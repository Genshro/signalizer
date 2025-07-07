package database

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type SupabaseClient struct {
	URL    string
	APIKey string
	Client *http.Client
}

type SupabaseError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
	Details string `json:"details"`
}

func NewSupabaseClient(url, apiKey string) (*SupabaseClient, error) {
	if url == "" || apiKey == "" {
		return nil, fmt.Errorf("supabase URL and API key are required")
	}

	return &SupabaseClient{
		URL:    url,
		APIKey: apiKey,
		Client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}, nil
}

func (s *SupabaseClient) makeRequest(method, endpoint string, body interface{}) ([]byte, error) {
	var reqBody io.Reader
	if body != nil {
		jsonData, err := json.Marshal(body)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal request body: %w", err)
		}
		reqBody = bytes.NewBuffer(jsonData)
	}

	url := fmt.Sprintf("%s/rest/v1/%s", s.URL, endpoint)
	req, err := http.NewRequest(method, url, reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Set headers
	req.Header.Set("apikey", s.APIKey)
	req.Header.Set("Authorization", "Bearer "+s.APIKey)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Prefer", "return=representation")

	resp, err := s.Client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to make request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response body: %w", err)
	}

	if resp.StatusCode >= 400 {
		var supabaseErr SupabaseError
		if err := json.Unmarshal(respBody, &supabaseErr); err == nil {
			return nil, fmt.Errorf("supabase error: %s - %s", supabaseErr.Code, supabaseErr.Message)
		}
		return nil, fmt.Errorf("supabase request failed with status %d: %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

func (s *SupabaseClient) Select(table string, query string) ([]byte, error) {
	endpoint := table
	if query != "" {
		endpoint += "?" + query
	}
	return s.makeRequest("GET", endpoint, nil)
}

func (s *SupabaseClient) Insert(table string, data interface{}) ([]byte, error) {
	return s.makeRequest("POST", table, data)
}

func (s *SupabaseClient) Update(table string, data interface{}, query string) ([]byte, error) {
	endpoint := table
	if query != "" {
		endpoint += "?" + query
	}
	return s.makeRequest("PATCH", endpoint, data)
}

func (s *SupabaseClient) Delete(table string, query string) ([]byte, error) {
	endpoint := table
	if query != "" {
		endpoint += "?" + query
	}
	return s.makeRequest("DELETE", endpoint, nil)
}

func (s *SupabaseClient) Upsert(table string, data interface{}) ([]byte, error) {
	endpoint := table
	return s.makeRequest("POST", endpoint, data)
}

// RPC calls for stored procedures
func (s *SupabaseClient) RPC(functionName string, params interface{}) ([]byte, error) {
	endpoint := fmt.Sprintf("rpc/%s", functionName)
	return s.makeRequest("POST", endpoint, params)
}
