package utils

import (
	"fmt"
	"log"
	"os"
	"strings"
	"time"
)

// PrintBenchmarkHeader prints the benchmark header with details about the test.
func PrintBenchmarkHeader(modelName string, inputTokens int, maxTokens int, latency float64) {
	banner :=
		`
##############################################################################################################################################
                                                          LLM API Throughput Benchmark
                                                    https://github.com/Yoosu-L/llmapibenchmark
                                                          Time：%s
##############################################################################################################################################`

	fmt.Printf(banner+"\n", time.Now().UTC().Format("2006-01-02 15:04:05 UTC+0"))
	fmt.Printf("Input Tokens: %d\n", inputTokens)
	fmt.Printf("Output Tokens: %d\n", maxTokens)
	fmt.Printf("Test Model: %s\n", modelName)
	fmt.Printf("Latency: %.2f ms\n\n", latency)
}

// SaveResultsToMD saves the benchmark results to a Markdown file.
func SaveResultsToMD(results [][]interface{}, modelName string, inputTokens int, maxTokens int, latency float64) {
	// sanitize modelName to create a safe filename (replace path separators)
	safeModelName := strings.ReplaceAll(modelName, "/", "_")
	safeModelName = strings.ReplaceAll(safeModelName, "\\", "_")
	safeModelName = strings.TrimSpace(safeModelName)
	if safeModelName == "" {
		safeModelName = "model"
	}
	filename := fmt.Sprintf("API_Throughput_%s.md", safeModelName)
	file, err := os.Create(filename)
	if err != nil {
		log.Printf("Error creating file: %v", err)
		return
	}
	defer file.Close()

	file.WriteString(fmt.Sprintf("```\nInput Tokens: %d\n", inputTokens))
	file.WriteString(fmt.Sprintf("Output Tokens: %d\n", maxTokens))
	file.WriteString(fmt.Sprintf("Test Model: %s\n", modelName))
	file.WriteString(fmt.Sprintf("Latency: %.2f ms\n```\n\n", latency))
	file.WriteString("| C | Gen Speed | Prompt TP | Total TP | Min TTFT | Avg TTFT | Med TTFT | P95 TTFT | P99 TTFT | StdDev | Success | Reqs | Duration |\n")
	file.WriteString("|---|-----------|-----------|----------|----------|----------|----------|----------|----------|--------|-------|------|----------|\n")

	for _, result := range results {
		concurrency := result[0].(int)
		generationSpeed := result[1].(float64)
		promptThroughput := result[2].(float64)
		totalThroughput := result[3].(float64)
		minTTFT := result[4].(float64)
		avgTTFT := result[5].(float64)
		medianTTFT := result[6].(float64)
		p95TTFT := result[7].(float64)
		p99TTFT := result[8].(float64)
		stdDevTTFT := result[9].(float64)
		successRate := result[10].(float64)
		successfulReqs := result[11].(int)
		duration := result[12].(float64)
		file.WriteString(fmt.Sprintf("| %2d | %9.2f | %9.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %6.2f | %5.2f%% | %4d | %8.2f |\n",
			concurrency,
			generationSpeed,
			promptThroughput,
			totalThroughput,
			minTTFT,
			avgTTFT,
			medianTTFT,
			p95TTFT,
			p99TTFT,
			stdDevTTFT,
			successRate*100,
			successfulReqs,
			duration,
		))
	}

	fmt.Printf("Results saved to: %s\n\n", filename)
}
