package main

import (
	"log"
	"os"

	"github.com/gin-gonic/gin"
	"github.com/joho/godotenv"
	"github.com/sirupsen/logrus"

	"signalizer-backend/internal/config"
	"signalizer-backend/internal/delivery/http"
	"signalizer-backend/pkg/database"
)

func main() {
	// Load environment variables
	if err := godotenv.Load(); err != nil {
		logrus.Warn("No .env file found")
	}

	// Initialize configuration
	cfg := config.New()

	// Initialize logger
	logrus.SetLevel(logrus.DebugLevel)
	logrus.SetFormatter(&logrus.JSONFormatter{})

	// Initialize Supabase client
	supabaseClient, err := database.NewSupabaseClient(cfg.Supabase.URL, cfg.Supabase.AnonKey)
	if err != nil {
		logrus.Fatal("Failed to initialize Supabase client: ", err)
	}

	// Initialize Redis client
	redisClient := database.NewRedisClient(cfg.Redis.URL, cfg.Redis.Password)

	// Initialize Gin router
	if cfg.App.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// Setup HTTP routes
	http.SetupRoutes(router, supabaseClient, redisClient, cfg)

	// Start server
	port := os.Getenv("API_PORT")
	if port == "" {
		port = "8080"
	}

	logrus.Infof("Starting server on port %s", port)
	if err := router.Run(":" + port); err != nil {
		log.Fatal("Failed to start server: ", err)
	}
}
