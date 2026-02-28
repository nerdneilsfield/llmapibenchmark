package utils

import (
	"math"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/Yoosu-L/llmapibenchmark/internal/api"

	"github.com/sashabaranov/go-openai"
	"github.com/schollz/progressbar/v3"
)

type SpeedMeasurement struct {
	BaseUrl        string
	ApiVersion     string
	ApiKey         string
	ModelName      string
	Prompt         string
	UseRandomInput bool
	NumWords       int
	MaxTokens      int
	Latency        float64
	Concurrency    int
}

type SpeedResult struct {
	Concurrency           int     `json:"concurrency" yaml:"concurrency"`
	GenerationSpeed       float64 `json:"generation_speed" yaml:"generation-speed"`
	PromptThroughput      float64 `json:"prompt_throughput" yaml:"prompt-throughput"`
	TotalThroughput       float64 `json:"total_throughput" yaml:"total-throughput"`
	MaxTtft               float64 `json:"max_ttft" yaml:"max-ttft"`
	MinTtft               float64 `json:"min_ttft" yaml:"min-ttft"`
	AvgTtft               float64 `json:"avg_ttft" yaml:"avg-ttft"`
	MedianTtft            float64 `json:"median_ttft" yaml:"median-ttft"`
	P95Ttft               float64 `json:"p95_ttft" yaml:"p95-ttft"`
	P99Ttft               float64 `json:"p99_ttft" yaml:"p99-ttft"`
	StdDevTtft            float64 `json:"stddev_ttft" yaml:"stddev-ttft"`
	SuccessRate           float64 `json:"success_rate" yaml:"success-rate"`
	SuccessfulRequests    int     `json:"successful_requests" yaml:"successful-requests"`
	FailedRequests        int     `json:"failed_requests" yaml:"failed-requests"`
	TotalPromptTokens     int     `json:"total_prompt_tokens" yaml:"total-prompt-tokens"`
	TotalCompletionTokens int     `json:"total_completion_tokens" yaml:"total-completion-tokens"`
	AvgPromptTokens       float64 `json:"avg_prompt_tokens" yaml:"avg-prompt-tokens"`
	AvgCompletionTokens   float64 `json:"avg_completion_tokens" yaml:"avg-completion-tokens"`
	Duration              float64 `json:"duration" yaml:"duration"`
}

func roundToTwoDecimals(f float64) float64 {
	return math.Round(f*100) / 100
}

func calculatePercentile(values []float64, percentile float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	index := int(math.Ceil(float64(len(sorted))*percentile)) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func calculateStdDev(values []float64, mean float64) float64 {
	if len(values) == 0 {
		return 0
	}
	var sum float64
	for _, v := range values {
		sum += math.Pow(v-mean, 2)
	}
	return math.Sqrt(sum / float64(len(values)))
}

// Run measures API generation throughput and TTFT.
func (setup *SpeedMeasurement) Run(bar *progressbar.ProgressBar) (SpeedResult, error) {
	config := openai.DefaultConfig(setup.ApiKey)
	config.BaseURL = setup.BaseUrl
	config.APIVersion = setup.ApiVersion
	client := openai.NewClientWithConfig(config)

	var wg sync.WaitGroup
	var responseTokens sync.Map
	var promptTokens sync.Map
	var ttfts sync.Map
	var successfulRequests atomic.Int32
	var failedRequests atomic.Int32

	start := time.Now()

	// Send requests concurrently (restored from debugging version)
	for i := 0; i < setup.Concurrency; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			var ttft float64
			var completionTokens, inputTokens int
			var err error
			if setup.UseRandomInput {
				ttft, completionTokens, inputTokens, err = api.AskOpenAiRandomInput(client, setup.ModelName, setup.NumWords, setup.MaxTokens, bar)
			} else {
				ttft, completionTokens, inputTokens, err = api.AskOpenAi(client, setup.ModelName, setup.Prompt, setup.MaxTokens, bar)
			}
			if err != nil {
				failedRequests.Add(1)
				return
			}
			successfulRequests.Add(1)
			ttfts.Store(index, ttft)
			responseTokens.Store(index, completionTokens)
			promptTokens.Store(index, inputTokens)
		}(i)
	}

	wg.Wait()
	duration := time.Since(start)

	// Calculate total tokens
	totalResponseTokens := 0
	responseTokens.Range(func(_, value interface{}) bool {
		totalResponseTokens += value.(int)
		return true
	})

	totalPromptTokens := 0
	promptTokens.Range(func(_, value interface{}) bool {
		totalPromptTokens += value.(int)
		return true
	})

	measurement := SpeedResult{}
	measurement.Concurrency = setup.Concurrency

	// Calculate success/failed requests
	measurement.SuccessfulRequests = int(successfulRequests.Load())
	measurement.FailedRequests = int(failedRequests.Load())

	// Calculate success rate
	totalRequests := setup.Concurrency
	if totalRequests > 0 {
		measurement.SuccessRate = float64(measurement.SuccessfulRequests) / float64(totalRequests)
	}

	// Collect TTFT values for statistics
	var ttftValues []float64
	ttfts.Range(func(_, value interface{}) bool {
		ttftValues = append(ttftValues, value.(float64))
		return true
	})

	// Calculate max, min, avg, median, P95, P99, stddev TTFT
	if len(ttftValues) > 0 {
		measurement.MaxTtft = ttftValues[0]
		measurement.MinTtft = ttftValues[0]
		var sumTtft float64
		for _, ttft := range ttftValues {
			sumTtft += ttft
			if ttft > measurement.MaxTtft {
				measurement.MaxTtft = ttft
			}
			if ttft < measurement.MinTtft {
				measurement.MinTtft = ttft
			}
		}
		measurement.AvgTtft = roundToTwoDecimals(sumTtft / float64(len(ttftValues)))
		measurement.MedianTtft = roundToTwoDecimals(calculatePercentile(ttftValues, 0.5))
		measurement.P95Ttft = roundToTwoDecimals(calculatePercentile(ttftValues, 0.95))
		measurement.P99Ttft = roundToTwoDecimals(calculatePercentile(ttftValues, 0.99))
		measurement.StdDevTtft = roundToTwoDecimals(calculateStdDev(ttftValues, measurement.AvgTtft))
	}

	measurement.MaxTtft = roundToTwoDecimals(measurement.MaxTtft)
	measurement.MinTtft = roundToTwoDecimals(measurement.MinTtft)
	measurement.Duration = roundToTwoDecimals(float64(duration.Seconds()))

	// Store total tokens
	measurement.TotalPromptTokens = totalPromptTokens
	measurement.TotalCompletionTokens = totalResponseTokens

	// Calculate average tokens per request
	if measurement.SuccessfulRequests > 0 {
		measurement.AvgPromptTokens = roundToTwoDecimals(float64(totalPromptTokens) / float64(measurement.SuccessfulRequests))
		measurement.AvgCompletionTokens = roundToTwoDecimals(float64(totalResponseTokens) / float64(measurement.SuccessfulRequests))
	}

	// Calculate speed (tokens/second)
	measurement.GenerationSpeed = roundToTwoDecimals(float64(totalResponseTokens) / (duration.Seconds() - setup.Latency/1000))

	// Calculate Prompt Throughput
	measurement.PromptThroughput = roundToTwoDecimals(float64(totalPromptTokens) / (measurement.MaxTtft - setup.Latency/1000))

	// Calculate Total Throughput (prompt + completion)
	measurement.TotalThroughput = roundToTwoDecimals(float64(totalPromptTokens+totalResponseTokens) / (duration.Seconds() - setup.Latency/1000))

	return measurement, nil
}
