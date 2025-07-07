package domain

import (
	"time"

	"github.com/google/uuid"
)

type SignalType string

const (
	SignalTypeBuy  SignalType = "BUY"
	SignalTypeSell SignalType = "SELL"
	SignalTypeHold SignalType = "HOLD"
)

type Signal struct {
	ID              uuid.UUID  `json:"id" db:"id"`
	CoinID          uuid.UUID  `json:"coin_id" db:"coin_id"`
	Coin            *Coin      `json:"coin,omitempty"`
	SignalType      SignalType `json:"signal_type" db:"signal_type"`
	BuyZoneMin      float64    `json:"buy_zone_min" db:"buy_zone_min"`
	BuyZoneMax      float64    `json:"buy_zone_max" db:"buy_zone_max"`
	Target1         float64    `json:"target_1" db:"target_1"`
	Target2         float64    `json:"target_2" db:"target_2"`
	Target3         float64    `json:"target_3" db:"target_3"`
	StopLoss        float64    `json:"stop_loss" db:"stop_loss"`
	ConfidenceScore float64    `json:"confidence_score" db:"confidence_score"`
	IndicatorUsed   string     `json:"indicator_used" db:"indicator_used"`
	Description     string     `json:"description" db:"description"`
	CreatedAt       time.Time  `json:"created_at" db:"created_at"`
	UpdatedAt       time.Time  `json:"updated_at" db:"updated_at"`
	ExpiresAt       *time.Time `json:"expires_at" db:"expires_at"`
	IsActive        bool       `json:"is_active" db:"is_active"`
	IsTriggered     bool       `json:"is_triggered" db:"is_triggered"`
	TriggeredAt     *time.Time `json:"triggered_at" db:"triggered_at"`
	TriggeredPrice  *float64   `json:"triggered_price" db:"triggered_price"`
}

type SignalPerformance struct {
	ID                uuid.UUID `json:"id" db:"id"`
	SignalID          uuid.UUID `json:"signal_id" db:"signal_id"`
	EntryPrice        float64   `json:"entry_price" db:"entry_price"`
	ExitPrice         *float64  `json:"exit_price" db:"exit_price"`
	ProfitLoss        *float64  `json:"profit_loss" db:"profit_loss"`
	ProfitLossPercent *float64  `json:"profit_loss_percent" db:"profit_loss_percent"`
	TargetReached     int       `json:"target_reached" db:"target_reached"` // 0=none, 1=target1, 2=target2, 3=target3
	IsStopLossHit     bool      `json:"is_stop_loss_hit" db:"is_stop_loss_hit"`
	Duration          *int64    `json:"duration" db:"duration"` // Duration in seconds
	CreatedAt         time.Time `json:"created_at" db:"created_at"`
	UpdatedAt         time.Time `json:"updated_at" db:"updated_at"`
}

type SignalAnalytics struct {
	TotalSignals      int     `json:"total_signals"`
	ActiveSignals     int     `json:"active_signals"`
	TriggeredSignals  int     `json:"triggered_signals"`
	SuccessRate       float64 `json:"success_rate"`
	AverageProfitLoss float64 `json:"average_profit_loss"`
	BestPerformance   float64 `json:"best_performance"`
	WorstPerformance  float64 `json:"worst_performance"`
	TotalProfit       float64 `json:"total_profit"`
	TotalLoss         float64 `json:"total_loss"`
}

type SignalFilter struct {
	CoinSymbol    string     `json:"coin_symbol"`
	SignalType    SignalType `json:"signal_type"`
	IsActive      *bool      `json:"is_active"`
	IsTriggered   *bool      `json:"is_triggered"`
	MinConfidence *float64   `json:"min_confidence"`
	MaxConfidence *float64   `json:"max_confidence"`
	CreatedAfter  *time.Time `json:"created_after"`
	CreatedBefore *time.Time `json:"created_before"`
	Limit         int        `json:"limit"`
	Offset        int        `json:"offset"`
}
