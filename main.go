package main

import (
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
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

type ChatResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Choices []struct {
		Index        int     `json:"index"`
		Message      Message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	} `json:"choices"`
}

var (
	sdBinPath      string
	diffusionModel string
	vaePath        string
	clipLPath      string
	t5xxlPath      string
	port           string
	mu             sync.Mutex
	outputDir      = "/home/konstantin.geyst/images"
)

func init() {
	flag.StringVar(&sdBinPath, "sd-bin", "", "Path to the sd binary")
	flag.StringVar(&diffusionModel, "diffusion-model", "", "Path to diffusion model")
	flag.StringVar(&vaePath, "vae", "", "Path to VAE file")
	flag.StringVar(&clipLPath, "clip_l", "", "Path to CLIP_L file")
	flag.StringVar(&t5xxlPath, "t5xxl", "", "Path to T5XXL file")
	flag.StringVar(&port, "port", "8080", "Port to run the web server on")
}

func extractPromptAndImage(messages []Message) (string, []byte, error) {
	var prompt string
	var imageData []byte

	for _, msg := range messages {
		if msg.Role != "user" {
			continue
		}
		for _, part := range msg.Content {
			switch part.Type {
			case "text":
				prompt += part.Text + " "
			case "image_url":
				if part.ImageURL != nil && strings.HasPrefix(part.ImageURL.URL, "data:image/") {
					idx := strings.Index(part.ImageURL.URL, "base64,")
					if idx == -1 {
						continue
					}
					raw := part.ImageURL.URL[idx+len("base64,"):]
					data, err := base64.StdEncoding.DecodeString(raw)
					if err != nil {
						return "", nil, fmt.Errorf("invalid base64 image: %w", err)
					}
					imageData = data
				}
			}
		}
	}
	return strings.TrimSpace(prompt), imageData, nil
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
	imgMarkdown := fmt.Sprintf("![output](/images/%s)", imageURL)

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
