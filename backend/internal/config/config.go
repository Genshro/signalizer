package config

import (
	"os"
	"strconv"
)

type Config struct {
	App      AppConfig
	Supabase SupabaseConfig
	Redis    RedisConfig
	API      APIConfig
	External ExternalConfig
}

type AppConfig struct {
	Environment string
	LogLevel    string
}

type SupabaseConfig struct {
	URL        string
	AnonKey    string
	ServiceKey string
	JWTSecret  string
}

type RedisConfig struct {
	URL      string
	Password string
}

type APIConfig struct {
	Port        string
	Host        string
	CORSOrigins string
}

type ExternalConfig struct {
	BinanceAPIKey    string
	BinanceSecretKey string
	CoinGeckoAPIKey  string
	FirebaseCredPath string
}

func New() *Config {
	return &Config{
		App: AppConfig{
			Environment: getEnv("APP_ENV", "development"),
			LogLevel:    getEnv("LOG_LEVEL", "debug"),
		},
		Supabase: SupabaseConfig{
			URL:        getEnv("SUPABASE_URL", ""),
			AnonKey:    getEnv("SUPABASE_ANON_KEY", ""),
			ServiceKey: getEnv("SUPABASE_SERVICE_KEY", ""),
			JWTSecret:  getEnv("JWT_SECRET", ""),
		},
		Redis: RedisConfig{
			URL:      getEnv("REDIS_URL", "redis://localhost:6379"),
			Password: getEnv("REDIS_PASSWORD", ""),
		},
		API: APIConfig{
			Port:        getEnv("API_PORT", "8080"),
			Host:        getEnv("API_HOST", "localhost"),
			CORSOrigins: getEnv("CORS_ORIGINS", "http://localhost:3000"),
		},
		External: ExternalConfig{
			BinanceAPIKey:    getEnv("BINANCE_API_KEY", ""),
			BinanceSecretKey: getEnv("BINANCE_SECRET_KEY", ""),
			CoinGeckoAPIKey:  getEnv("COINGECKO_API_KEY", ""),
			FirebaseCredPath: getEnv("FIREBASE_CREDENTIALS_PATH", "./firebase-credentials.json"),
		},
	}
}

func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

func getEnvAsInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}

func getEnvAsBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		if boolValue, err := strconv.ParseBool(value); err == nil {
			return boolValue
		}
	}
	return defaultValue
}
