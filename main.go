package main

import (
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

type ContentPart struct {
	Type     string     `json:"type"`
	Text     string     `json:"text,omitempty"`
	ImageURL *ImagePart `json:"image_url,omitempty"`
}

type ImagePart struct {
	URL string `json:"url"`
}

type Message struct {
	Role    string        `json:"role"`
	Content []ContentPart `json:"content"`
}

func (m *Message) UnmarshalJSON(data []byte) error {
	type Alias Message
	aux := &struct {
		Content json.RawMessage `json:"content"`
		*Alias
	}{
		Alias: (*Alias)(m),
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	var parts []ContentPart
	if err := json.Unmarshal(aux.Content, &parts); err == nil {
		m.Content = parts
		return nil
	}

	var text string
	if err := json.Unmarshal(aux.Content, &text); err == nil {
		m.Content = []ContentPart{{Type: "text", Text: text}}
		return nil
	}

	return fmt.Errorf("invalid content format in Message")
}

type ChatRequest struct {
	Model    string    `json:"model"`
	Messages []Message `json:"messages"`
}

var (
	sdBinPath      string
	diffusionModel string
	vaePath        string
	clipLPath      string
	t5xxlPath      string
	port           string
	mu             sync.Mutex
	outputDir      string
)

func init() {
	flag.StringVar(&sdBinPath, "sd-bin", "", "Path to the sd binary")
	flag.StringVar(&diffusionModel, "diffusion-model", "", "Path to diffusion model")
	flag.StringVar(&vaePath, "vae", "", "Path to VAE file")
	flag.StringVar(&clipLPath, "clip_l", "", "Path to CLIP_L file")
	flag.StringVar(&t5xxlPath, "t5xxl", "", "Path to T5XXL file")
	flag.StringVar(&port, "port", "8080", "Port to run the web server on")
	flag.StringVar(&outputDir, "output-dir", "", "Directory to save generated images")
}

func extractPromptAndImage(messages []Message) (string, []byte, error) {
	var lastText string
	var lastImageData []byte
	var lastImageURL string
	imagePattern := regexp.MustCompile(`(?:https?:\/\/\S+|\b\/[^ \n\t\r]+)\.png\b`)

	for _, msg := range messages {
		for _, part := range msg.Content {
			switch part.Type {
			case "text":
				if msg.Role == "user" {
					lastText = part.Text
				}

				// Search for .png URL in text
				matches := imagePattern.FindAllString(part.Text, -1)
				if len(matches) > 0 {
					lastImageURL = matches[len(matches)-1]
				}

			case "image_url":
				if part.ImageURL != nil {
					urlStr := part.ImageURL.URL

					if strings.HasPrefix(urlStr, "data:image/") {
						idx := strings.Index(urlStr, "base64,")
						if idx == -1 {
							continue
						}
						raw := urlStr[idx+len("base64,"):]
						data, err := base64.StdEncoding.DecodeString(raw)
						if err != nil {
							log.Printf("Invalid base64 image skipped: %v", err)
							continue
						}
						lastImageData = data
					} else if strings.HasSuffix(urlStr, ".png") {
						lastImageURL = urlStr
					}
				}
			}
		}
	}

	// If no image data was found, but a URL/relative path was:
	if len(lastImageData) == 0 && lastImageURL != "" {
		finalURL := lastImageURL
		if strings.HasPrefix(finalURL, "/") {
			finalURL = "https://web.ai.ispring.lan/generated" + finalURL
		}
		// Validate URL
		if u, err := url.Parse(finalURL); err == nil && u.Scheme != "" {
			// Custom client that skips cert verification
			tr := &http.Transport{
				TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
			}
			client := &http.Client{Transport: tr}

			resp, err := client.Get(finalURL)
			if err != nil {
				return strings.TrimSpace(lastText), nil, fmt.Errorf("failed to fetch image from URL: %w", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != http.StatusOK {
				return strings.TrimSpace(lastText), nil, fmt.Errorf("image URL returned status: %s", resp.Status)
			}

			imgData, err := io.ReadAll(resp.Body)
			if err != nil {
				return strings.TrimSpace(lastText), nil, fmt.Errorf("failed to read image data from response: %w", err)
			}
			lastImageData = imgData
		}
	}

	return strings.TrimSpace(lastText), lastImageData, nil
}

func handleChatCompletion(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	defer mu.Unlock()

	ctx := r.Context()

	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "Failed to read request body", http.StatusInternalServerError)
		log.Printf("Body read error: %v\n", err)
		return
	}

	fmt.Println("Raw JSON request:")
	fmt.Println(string(bodyBytes))

	var req ChatRequest
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		log.Printf("Request decode error: %v\n", err)
		return
	}

	prompt, imageData, err := extractPromptAndImage(req.Messages)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		log.Printf("Prompt/Image extraction error: %v\n", err)
		return
	}

	fmt.Println("Prompt:", prompt)
	if len(imageData) > 0 {
		fmt.Printf("Image Data: %d bytes\n", len(imageData))
	} else {
		fmt.Println("Image Data: <none>")
	}

	if prompt == "" {
		http.Error(w, "No user prompt provided", http.StatusBadRequest)
		log.Println("No user prompt provided")
		return
	}

	args := []string{
		"--diffusion-model", diffusionModel,
		"--vae", vaePath,
		"--clip_l", clipLPath,
		"--t5xxl", t5xxlPath,
		"-p", prompt,
		"--cfg-scale", "1.0",
		"--sampling-method", "euler",
		"--seed", "-1",
		"-v",
	}

	if len(imageData) > 0 {
		if err := os.WriteFile("input.png", imageData, 0644); err != nil {
			http.Error(w, "Failed to write input image", http.StatusInternalServerError)
			return
		}
		defer os.Remove("input.png")
		args = append(args, "-M", "edit", "-r", "input.png")
	}

	cmd := exec.CommandContext(ctx, sdBinPath, args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		log.Printf("Command failed: %v", err)
		http.Error(w, "Failed to run model", http.StatusInternalServerError)
		return
	}

	outputPath := filepath.Join(outputDir, fmt.Sprintf("output_%d.png", time.Now().UnixNano()))
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		http.Error(w, "Failed to create output directory", http.StatusInternalServerError)
		return
	}

	imgData, err := os.ReadFile("output.png")
	if err != nil {
		http.Error(w, "Failed to read output.png", http.StatusInternalServerError)
		return
	}
	if err := os.WriteFile(outputPath, imgData, 0644); err != nil {
		http.Error(w, "Failed to save generated image", http.StatusInternalServerError)
		return
	}

	imageURL := filepath.Base(outputPath) // e.g., output_123456.png
	imgMarkdown := fmt.Sprintf("![output](/generated/%s)", imageURL)

	response := map[string]interface{}{
		"id":      "chatcmpl-mockid",
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   req.Model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]string{
					"role":    "assistant",
					"content": imgMarkdown,
				},
				"finish_reason": "stop",
			},
		},
	}

	respBytes, err := json.MarshalIndent(response, "", "  ")
	if err != nil {
		log.Printf("Failed to marshal response: %v", err)
		http.Error(w, "Internal server error", http.StatusInternalServerError)
		return
	}

	fmt.Println("Response JSON:")
	fmt.Println(string(respBytes))

	w.Header().Set("Content-Type", "application/json")
	w.Write(respBytes)
}

func main() {
	flag.Parse()

	if diffusionModel == "" || vaePath == "" || clipLPath == "" || t5xxlPath == "" {
		log.Fatal("All model component paths must be provided via flags.")
	}

	http.HandleFunc("/v1/chat/completions", handleChatCompletion)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = io.WriteString(w, "OK")
	})

	addr := fmt.Sprintf(":%s", port)
	fmt.Printf("Server running on http://localhost%s\n", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}
