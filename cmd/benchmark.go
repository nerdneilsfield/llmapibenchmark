package main

import (
	"fmt"
	"os"

	"github.com/Yoosu-L/llmapibenchmark/internal/utils"
	"github.com/schollz/progressbar/v3"
)

func (benchmark *Benchmark) runCli() error {
	// Test latency
	latency, err := utils.MeasureLatency(benchmark.BaseURL, 5)
	if err != nil {
		return fmt.Errorf("latency test error: %v", err)
	}

	// Print benchmark header
	utils.PrintBenchmarkHeader(benchmark.ModelName, benchmark.InputTokens, benchmark.MaxTokens, latency)

	// Print table header
	fmt.Println("| C | Gen Speed | Prompt TP | Total TP | Min TTFT | Avg TTFT | Med TTFT | P95 TTFT | P99 TTFT | StdDev | Success | Reqs | Duration |")
	fmt.Println("|---|-----------|-----------|----------|----------|----------|----------|----------|----------|--------|-------|------|----------|")

	// Test each concurrency level and print results
	var results [][]interface{}
	for _, concurrency := range benchmark.ConcurrencyLevels {
		result, err := benchmark.measureSpeed(latency, concurrency, true)
		if err != nil {
			return fmt.Errorf("concurrency %d: %v", concurrency, err)
		}

		// Print current results
		fmt.Printf("| %2d | %9.2f | %9.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %8.2f | %6.2f | %5.2f%% | %4d | %8.2f |\n",
			concurrency,
			result.GenerationSpeed,
			result.PromptThroughput,
			result.TotalThroughput,
			result.MinTtft,
			result.AvgTtft,
			result.MedianTtft,
			result.P95Ttft,
			result.P99Ttft,
			result.StdDevTtft,
			result.SuccessRate*100,
			result.SuccessfulRequests,
			result.Duration,
		)

		// Save results for later
		results = append(results, []interface{}{
			concurrency,
			result.GenerationSpeed,
			result.PromptThroughput,
			result.TotalThroughput,
			result.MinTtft,
			result.AvgTtft,
			result.MedianTtft,
			result.P95Ttft,
			result.P99Ttft,
			result.StdDevTtft,
			result.SuccessRate,
			result.SuccessfulRequests,
			result.Duration,
		})
	}

	fmt.Println("|---|-----------|-----------|----------|----------|----------|----------|----------|----------|--------|-------|------|----------|")
	fmt.Println("\n====================================================================================================")

	// Save results to Markdown
	utils.SaveResultsToMD(results, benchmark.ModelName, benchmark.InputTokens, benchmark.MaxTokens, latency)

	return nil
}

func (benchmark *Benchmark) run() (BenchmarkResult, error) {
	result := BenchmarkResult{}
	result.ModelName = benchmark.ModelName
	result.InputTokens = benchmark.InputTokens
	result.MaxTokens = benchmark.MaxTokens

	// Test latency
	latency, err := utils.MeasureLatency(benchmark.BaseURL, 5)
	if err != nil {
		return result, fmt.Errorf("error testing latency: %v", err)
	}
	result.Latency = latency

	for _, concurrency := range benchmark.ConcurrencyLevels {
		measurement, err := benchmark.measureSpeed(latency, concurrency, false)
		if err != nil {
			return result, fmt.Errorf("concurrency %d: %v", concurrency, err)
		}

		result.Results = append(result.Results, measurement)
	}

	return result, nil
}

func (benchmark *Benchmark) measureSpeed(latency float64, concurrency int, clearProgress bool) (utils.SpeedResult, error) {

	// Create a progress bar for this specific concurrency level
	expectedTokens := concurrency * benchmark.MaxTokens
	bar := progressbar.NewOptions(expectedTokens,
		progressbar.OptionSetWriter(os.Stderr),
		progressbar.OptionSetDescription(fmt.Sprintf("Concurrency %d", concurrency)),
		progressbar.OptionSetWidth(40),
		progressbar.OptionShowCount(),
		progressbar.OptionShowIts(),
		progressbar.OptionSetItsString("tokens"),
		progressbar.OptionSpinnerType(14),
		progressbar.OptionSetRenderBlankState(true),
	)

	speedMeasurement := utils.SpeedMeasurement{
		BaseUrl:     benchmark.BaseURL,
		ApiVersion:  benchmark.ApiVersion,
		ApiKey:      benchmark.ApiKey,
		ModelName:   benchmark.ModelName,
		Prompt:      benchmark.Prompt,
		NumWords:    benchmark.NumWords,
		MaxTokens:   benchmark.MaxTokens,
		Latency:     latency,
		Concurrency: concurrency,
	}
	if benchmark.UseRandomInput {
		speedMeasurement.UseRandomInput = true
	}

	result, err := speedMeasurement.Run(bar)
	if err != nil {
		return result, fmt.Errorf("measurement error: %v", err)
	}

	bar.Finish()
	if clearProgress {
		bar.Clear()
	} else {
		fmt.Fprintf(os.Stderr, "\n")
	}
	bar.Close()

	return result, nil
}
