package domain

import (
	"time"

	"github.com/google/uuid"
)

type User struct {
	ID        uuid.UUID `json:"id" db:"id"`
	Email     string    `json:"email" db:"email"`
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

type UserProfile struct {
	ID                  uuid.UUID `json:"id" db:"id"`
	UserID              uuid.UUID `json:"user_id" db:"user_id"`
	Username            string    `json:"username" db:"username"`
	FullName            string    `json:"full_name" db:"full_name"`
	AvatarURL           string    `json:"avatar_url" db:"avatar_url"`
	PreferredCurrency   string    `json:"preferred_currency" db:"preferred_currency"`
	NotificationEnabled bool      `json:"notification_enabled" db:"notification_enabled"`
	ThemePreference     string    `json:"theme_preference" db:"theme_preference"`
	Language            string    `json:"language" db:"language"`
	TimeZone            string    `json:"timezone" db:"timezone"`
	CreatedAt           time.Time `json:"created_at" db:"created_at"`
	UpdatedAt           time.Time `json:"updated_at" db:"updated_at"`
}

type UserFavorite struct {
	ID        uuid.UUID `json:"id" db:"id"`
	UserID    uuid.UUID `json:"user_id" db:"user_id"`
	CoinID    uuid.UUID `json:"coin_id" db:"coin_id"`
	Coin      *Coin     `json:"coin,omitempty"`
	CreatedAt time.Time `json:"created_at" db:"created_at"`
}

type UserNotificationSettings struct {
	ID                    uuid.UUID `json:"id" db:"id"`
	UserID                uuid.UUID `json:"user_id" db:"user_id"`
	SignalNotifications   bool      `json:"signal_notifications" db:"signal_notifications"`
	PriceAlerts           bool      `json:"price_alerts" db:"price_alerts"`
	NewsUpdates           bool      `json:"news_updates" db:"news_updates"`
	MarketUpdates         bool      `json:"market_updates" db:"market_updates"`
	EmailNotifications    bool      `json:"email_notifications" db:"email_notifications"`
	PushNotifications     bool      `json:"push_notifications" db:"push_notifications"`
	SMSNotifications      bool      `json:"sms_notifications" db:"sms_notifications"`
	NotificationFrequency string    `json:"notification_frequency" db:"notification_frequency"` // immediate, hourly, daily
	CreatedAt             time.Time `json:"created_at" db:"created_at"`
	UpdatedAt             time.Time `json:"updated_at" db:"updated_at"`
}

type UserSession struct {
	ID        uuid.UUID `json:"id" db:"id"`
	UserID    uuid.UUID `json:"user_id" db:"user_id"`
	Token     string    `json:"token" db:"token"`
	DeviceID  string    `json:"device_id" db:"device_id"`
	UserAgent string    `json:"user_agent" db:"user_agent"`
	IPAddress string    `json:"ip_address" db:"ip_address"`
	IsActive  bool      `json:"is_active" db:"is_active"`
	ExpiresAt time.Time `json:"expires_at" db:"expires_at"`
	CreatedAt time.Time `json:"created_at" db:"created_at"`
	UpdatedAt time.Time `json:"updated_at" db:"updated_at"`
}

type UserActivity struct {
	ID          uuid.UUID `json:"id" db:"id"`
	UserID      uuid.UUID `json:"user_id" db:"user_id"`
	Action      string    `json:"action" db:"action"`
	Description string    `json:"description" db:"description"`
	IPAddress   string    `json:"ip_address" db:"ip_address"`
	UserAgent   string    `json:"user_agent" db:"user_agent"`
	CreatedAt   time.Time `json:"created_at" db:"created_at"`
}
