package domain

import (
	"time"

	"github.com/google/uuid"
)

type Coin struct {
	ID             uuid.UUID `json:"id" db:"id"`
	Symbol         string    `json:"symbol" db:"symbol"`
	Name           string    `json:"name" db:"name"`
	CurrentPrice   float64   `json:"current_price" db:"current_price"`
	MarketCap      int64     `json:"market_cap" db:"market_cap"`
	Volume24h      int64     `json:"volume_24h" db:"volume_24h"`
	PriceChange24h float64   `json:"price_change_24h" db:"price_change_24h"`
	LastUpdated    time.Time `json:"last_updated" db:"last_updated"`
	IsActive       bool      `json:"is_active" db:"is_active"`
	CreatedAt      time.Time `json:"created_at" db:"created_at"`
	UpdatedAt      time.Time `json:"updated_at" db:"updated_at"`
}

type CoinPriceHistory struct {
	ID        uuid.UUID `json:"id" db:"id"`
	CoinID    uuid.UUID `json:"coin_id" db:"coin_id"`
	Price     float64   `json:"price" db:"price"`
	Volume    int64     `json:"volume" db:"volume"`
	Timestamp time.Time `json:"timestamp" db:"timestamp"`
}

type CoinMarketData struct {
	Symbol                    string    `json:"symbol"`
	Name                      string    `json:"name"`
	CurrentPrice              float64   `json:"current_price"`
	MarketCap                 int64     `json:"market_cap"`
	MarketCapRank             int       `json:"market_cap_rank"`
	FullyDilutedValuation     int64     `json:"fully_diluted_valuation"`
	TotalVolume               int64     `json:"total_volume"`
	High24h                   float64   `json:"high_24h"`
	Low24h                    float64   `json:"low_24h"`
	PriceChange24h            float64   `json:"price_change_24h"`
	PriceChangePercent24h     float64   `json:"price_change_percentage_24h"`
	MarketCapChange24h        int64     `json:"market_cap_change_24h"`
	MarketCapChangePercent24h float64   `json:"market_cap_change_percentage_24h"`
	CirculatingSupply         float64   `json:"circulating_supply"`
	TotalSupply               float64   `json:"total_supply"`
	MaxSupply                 float64   `json:"max_supply"`
	ATH                       float64   `json:"ath"`
	ATHChangePercent          float64   `json:"ath_change_percentage"`
	ATHDate                   time.Time `json:"ath_date"`
	ATL                       float64   `json:"atl"`
	ATLChangePercent          float64   `json:"atl_change_percentage"`
	ATLDate                   time.Time `json:"atl_date"`
	LastUpdated               time.Time `json:"last_updated"`
}
