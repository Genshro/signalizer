//! On-Balance Volume (OBV) Indicator
//!
//! OBV is a momentum indicator that uses volume flow to predict changes in stock price.
//! It combines price and volume to show how money may be flowing into or out of a security.
//! Rising OBV reflects positive volume pressure that can lead to higher prices.

use crate::types::{
    IndicatorResult, OhlcData, Signal, SignalStrength, TechnicalAnalysisError, Timeframe,
};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};

/// OBV Configuration
#[derive(Debug, Clone)]
pub struct ObvConfig {
    /// Period for signal generation (default: 14)
    pub signal_period: usize,
    /// Minimum volume threshold for calculation
    pub min_volume_threshold: Decimal,
    /// Enable divergence detection
    pub enable_divergence: bool,
}

impl Default for ObvConfig {
    fn default() -> Self {
        Self {
            signal_period: 14,
            min_volume_threshold: dec!(0.0),
            enable_divergence: true,
        }
    }
}

/// OBV calculation result
#[derive(Debug, Clone)]
pub struct ObvResult {
    /// Current OBV value
    pub obv_value: Decimal,
    /// OBV trend direction
    pub trend: ObvTrend,
    /// Signal strength
    pub signal_strength: SignalStrength,
    /// Volume pressure
    pub volume_pressure: VolumePressure,
    /// Divergence detection (if enabled)
    pub divergence: Option<ObvDivergence>,
}

/// OBV trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObvTrend {
    /// Strong uptrend
    StrongUp,
    /// Weak uptrend
    WeakUp,
    /// Neutral/sideways
    Neutral,
    /// Weak downtrend
    WeakDown,
    /// Strong downtrend
    StrongDown,
}

/// Volume pressure types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumePressure {
    /// Strong buying pressure
    StrongBuying,
    /// Moderate buying pressure
    ModerateBuying,
    /// Neutral pressure
    Neutral,
    /// Moderate selling pressure
    ModerateSelling,
    /// Strong selling pressure
    StrongSelling,
}

/// OBV divergence detection
#[derive(Debug, Clone)]
pub struct ObvDivergence {
    /// Divergence type
    pub divergence_type: DivergenceType,
    /// Divergence strength (0.0 to 1.0)
    pub strength: Decimal,
    /// Number of periods in divergence
    pub periods: usize,
}

/// Divergence types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceType {
    /// Bullish divergence (price down, OBV up)
    Bullish,
    /// Bearish divergence (price up, OBV down)
    Bearish,
    /// No divergence
    None,
}

/// On-Balance Volume indicator
#[derive(Debug, Clone)]
pub struct OnBalanceVolume {
    /// Configuration
    config: ObvConfig,
    /// Current OBV value
    current_obv: Decimal,
    /// Previous close price
    previous_close: Option<Decimal>,
    /// OBV history for trend analysis
    obv_history: VecDeque<Decimal>,
    /// Price history for divergence detection
    price_history: VecDeque<Decimal>,
    /// Volume history
    volume_history: VecDeque<Decimal>,
    /// Timestamps for tracking
    timestamps: VecDeque<DateTime<Utc>>,
    /// Is indicator ready
    is_ready: bool,
}

impl OnBalanceVolume {
    /// Create new OBV indicator
    pub fn new(config: ObvConfig) -> Result<Self, TechnicalAnalysisError> {
        if config.signal_period == 0 {
            return Err(TechnicalAnalysisError::invalid_config(
                "Signal period cannot be zero",
            ));
        }

        Ok(Self {
            config,
            current_obv: dec!(0),
            previous_close: None,
            obv_history: VecDeque::new(),
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            timestamps: VecDeque::new(),
            is_ready: false,
        })
    }

    /// Update OBV with new OHLC data
    pub fn update(
        &mut self,
        ohlc: &OhlcData,
    ) -> Result<Option<IndicatorResult>, TechnicalAnalysisError> {
        // Skip if volume is below threshold
        if ohlc.volume < self.config.min_volume_threshold {
            return Ok(None);
        }

        // Calculate OBV
        if let Some(prev_close) = self.previous_close {
            if ohlc.close > prev_close {
                // Price up: add volume
                self.current_obv += ohlc.volume;
            } else if ohlc.close < prev_close {
                // Price down: subtract volume
                self.current_obv -= ohlc.volume;
            }
            // Price unchanged: OBV unchanged
        } else {
            // First data point
            self.current_obv = ohlc.volume;
        }

        // Update history
        self.obv_history.push_back(self.current_obv);
        self.price_history.push_back(ohlc.close);
        self.volume_history.push_back(ohlc.volume);
        self.timestamps.push_back(ohlc.timestamp);

        // Keep limited history
        let max_history = self.config.signal_period * 3;
        if self.obv_history.len() > max_history {
            self.obv_history.pop_front();
            self.price_history.pop_front();
            self.volume_history.pop_front();
            self.timestamps.pop_front();
        }

        // Update previous close
        self.previous_close = Some(ohlc.close);

        // Check if ready
        if self.obv_history.len() >= self.config.signal_period {
            self.is_ready = true;
        }

        if !self.is_ready {
            return Ok(None);
        }

        // Generate result
        let result = self.generate_result(ohlc)?;
        Ok(Some(result))
    }

    /// Generate OBV result
    fn generate_result(&self, ohlc: &OhlcData) -> Result<IndicatorResult, TechnicalAnalysisError> {
        let trend = self.calculate_trend();
        let volume_pressure = self.calculate_volume_pressure();
        let signal_strength = self.calculate_signal_strength();
        let divergence_data = if self.config.enable_divergence {
            self.detect_divergence()
        } else {
            None
        };

        let obv_result = ObvResult {
            obv_value: self.current_obv,
            trend,
            signal_strength,
            volume_pressure,
            divergence: divergence_data.clone(),
        };

        // Generate signal
        let signal = self.generate_signal_from_result(&obv_result);

        // Create additional values
        let mut additional_values = HashMap::new();
        additional_values.insert("trend".to_string(), Decimal::from(trend as u8));
        additional_values.insert(
            "volume_pressure".to_string(),
            Decimal::from(volume_pressure as u8),
        );
        additional_values.insert(
            "signal_strength".to_string(),
            Decimal::from(signal_strength as u8),
        );

        if let Some(ref div) = divergence_data {
            additional_values.insert(
                "divergence_type".to_string(),
                Decimal::from(div.divergence_type as u8),
            );
            additional_values.insert("divergence_strength".to_string(), div.strength);
        }

        let mut result = IndicatorResult::new(self.current_obv, ohlc.timestamp, "OBV".to_string());

        result.additional_values = Some(additional_values);
        result.signal = Some(signal);

        Ok(result)
    }

    /// Calculate OBV trend
    fn calculate_trend(&self) -> ObvTrend {
        if self.obv_history.len() < self.config.signal_period {
            return ObvTrend::Neutral;
        }

        let recent_values: Vec<Decimal> = self
            .obv_history
            .iter()
            .rev()
            .take(self.config.signal_period)
            .cloned()
            .collect();

        let first_half = &recent_values[recent_values.len() / 2..];
        let second_half = &recent_values[..recent_values.len() / 2];

        let first_avg = first_half.iter().sum::<Decimal>() / Decimal::from(first_half.len());
        let second_avg = second_half.iter().sum::<Decimal>() / Decimal::from(second_half.len());

        let change_ratio = if first_avg != dec!(0) {
            (second_avg - first_avg) / first_avg.abs()
        } else {
            dec!(0)
        };

        if change_ratio > dec!(0.1) {
            ObvTrend::StrongUp
        } else if change_ratio > dec!(0.05) {
            ObvTrend::WeakUp
        } else if change_ratio < dec!(-0.1) {
            ObvTrend::StrongDown
        } else if change_ratio < dec!(-0.05) {
            ObvTrend::WeakDown
        } else {
            ObvTrend::Neutral
        }
    }

    /// Calculate volume pressure
    fn calculate_volume_pressure(&self) -> VolumePressure {
        if self.volume_history.len() < 3 {
            return VolumePressure::Neutral;
        }

        let recent_volumes: Vec<Decimal> =
            self.volume_history.iter().rev().take(3).cloned().collect();

        let recent_prices: Vec<Decimal> =
            self.price_history.iter().rev().take(3).cloned().collect();

        let mut buying_volume = dec!(0);
        let mut selling_volume = dec!(0);

        for i in 1..recent_volumes.len() {
            if recent_prices[i - 1] > recent_prices[i] {
                buying_volume += recent_volumes[i];
            } else if recent_prices[i - 1] < recent_prices[i] {
                selling_volume += recent_volumes[i];
            }
        }

        let total_volume = buying_volume + selling_volume;
        if total_volume == dec!(0) {
            return VolumePressure::Neutral;
        }

        let buying_ratio = buying_volume / total_volume;

        if buying_ratio > dec!(0.7) {
            VolumePressure::StrongBuying
        } else if buying_ratio > dec!(0.6) {
            VolumePressure::ModerateBuying
        } else if buying_ratio < dec!(0.3) {
            VolumePressure::StrongSelling
        } else if buying_ratio < dec!(0.4) {
            VolumePressure::ModerateSelling
        } else {
            VolumePressure::Neutral
        }
    }

    /// Calculate signal strength
    fn calculate_signal_strength(&self) -> SignalStrength {
        let trend_strength = match self.calculate_trend() {
            ObvTrend::StrongUp | ObvTrend::StrongDown => 3,
            ObvTrend::WeakUp | ObvTrend::WeakDown => 2,
            ObvTrend::Neutral => 1,
        };

        let volume_strength = match self.calculate_volume_pressure() {
            VolumePressure::StrongBuying | VolumePressure::StrongSelling => 3,
            VolumePressure::ModerateBuying | VolumePressure::ModerateSelling => 2,
            VolumePressure::Neutral => 1,
        };

        let combined_strength = (trend_strength + volume_strength) / 2;

        match combined_strength {
            3 => SignalStrength::Strong,
            2 => SignalStrength::Moderate,
            _ => SignalStrength::Weak,
        }
    }

    /// Detect divergence between price and OBV
    fn detect_divergence(&self) -> Option<ObvDivergence> {
        if self.obv_history.len() < self.config.signal_period * 2
            || self.price_history.len() < self.config.signal_period * 2
        {
            return None;
        }

        let lookback = self.config.signal_period;
        let recent_obv: Vec<Decimal> = self
            .obv_history
            .iter()
            .rev()
            .take(lookback)
            .cloned()
            .collect();

        let recent_prices: Vec<Decimal> = self
            .price_history
            .iter()
            .rev()
            .take(lookback)
            .cloned()
            .collect();

        // Calculate trends
        let obv_trend = self.calculate_trend_direction(&recent_obv);
        let price_trend = self.calculate_trend_direction(&recent_prices);

        let divergence_type = match (price_trend, obv_trend) {
            (TrendDir::Down, TrendDir::Up) => DivergenceType::Bullish,
            (TrendDir::Up, TrendDir::Down) => DivergenceType::Bearish,
            _ => DivergenceType::None,
        };

        if divergence_type != DivergenceType::None {
            let strength = self.calculate_divergence_strength(&recent_prices, &recent_obv);
            Some(ObvDivergence {
                divergence_type,
                strength,
                periods: lookback,
            })
        } else {
            None
        }
    }

    /// Calculate trend direction for divergence
    fn calculate_trend_direction(&self, values: &[Decimal]) -> TrendDir {
        if values.len() < 2 {
            return TrendDir::Sideways;
        }

        let first = values[values.len() - 1];
        let last = values[0];

        let change = (last - first) / first.abs();

        if change > dec!(0.02) {
            TrendDir::Up
        } else if change < dec!(-0.02) {
            TrendDir::Down
        } else {
            TrendDir::Sideways
        }
    }

    /// Calculate divergence strength
    fn calculate_divergence_strength(&self, prices: &[Decimal], obv_values: &[Decimal]) -> Decimal {
        if prices.len() != obv_values.len() || prices.len() < 2 {
            return dec!(0);
        }

        let price_change = (prices[0] - prices[prices.len() - 1]) / prices[prices.len() - 1].abs();
        let obv_change = (obv_values[0] - obv_values[obv_values.len() - 1])
            / obv_values[obv_values.len() - 1].abs();

        // Strength is based on how much the trends diverge
        (price_change - obv_change).abs().min(dec!(1.0))
    }

    /// Generate signal from OBV result
    fn generate_signal_from_result(&self, result: &ObvResult) -> Signal {
        match (result.trend, result.volume_pressure) {
            (ObvTrend::StrongUp, VolumePressure::StrongBuying) => Signal::StrongBuy,
            (ObvTrend::WeakUp, VolumePressure::ModerateBuying) => Signal::Buy,
            (ObvTrend::StrongDown, VolumePressure::StrongSelling) => Signal::StrongSell,
            (ObvTrend::WeakDown, VolumePressure::ModerateSelling) => Signal::Sell,
            _ => Signal::Neutral,
        }
    }

    /// Generate signal for external use
    pub fn generate_signal(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<crate::types::SignalData>, TechnicalAnalysisError> {
        if !self.is_ready {
            return Ok(None);
        }

        let trend = self.calculate_trend();
        let volume_pressure = self.calculate_volume_pressure();
        let signal_strength = self.calculate_signal_strength();

        let signal = match (trend, volume_pressure) {
            (ObvTrend::StrongUp, VolumePressure::StrongBuying) => Signal::StrongBuy,
            (ObvTrend::WeakUp, VolumePressure::ModerateBuying) => Signal::Buy,
            (ObvTrend::StrongDown, VolumePressure::StrongSelling) => Signal::StrongSell,
            (ObvTrend::WeakDown, VolumePressure::ModerateSelling) => Signal::Sell,
            _ => Signal::Neutral,
        };

        // Create metadata
        let metadata = serde_json::json!({
            "obv_value": self.current_obv.to_string(),
            "trend": format!("{:?}", trend),
            "volume_pressure": format!("{:?}", volume_pressure)
        });

        let signal_data = crate::types::SignalData {
            signal,
            strength: signal_strength,
            confidence: dec!(0.7), // Default confidence
            timestamp,
            source: "OBV".to_string(),
            metadata: Some(metadata),
            price: self.price_history.back().copied(),
            volume: self.volume_history.back().copied(),
            timeframe: Some(Timeframe::H1),
            duration: Some(chrono::Duration::hours(1)),
            stop_loss: None,
            take_profit: None,
            risk_reward_ratio: None,
        };

        Ok(Some(signal_data))
    }

    /// Check if OBV shows bullish signal
    pub fn is_bullish(&self) -> bool {
        matches!(
            self.calculate_trend(),
            ObvTrend::StrongUp | ObvTrend::WeakUp
        )
    }

    /// Check if OBV shows bearish signal
    pub fn is_bearish(&self) -> bool {
        matches!(
            self.calculate_trend(),
            ObvTrend::StrongDown | ObvTrend::WeakDown
        )
    }

    /// Get current OBV value
    pub fn get_obv_value(&self) -> Decimal {
        self.current_obv
    }

    /// Check if indicator is ready
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    /// Reset indicator
    pub fn reset(&mut self) {
        self.current_obv = dec!(0);
        self.previous_close = None;
        self.obv_history.clear();
        self.price_history.clear();
        self.volume_history.clear();
        self.timestamps.clear();
        self.is_ready = false;
    }
}

impl Default for OnBalanceVolume {
    fn default() -> Self {
        Self {
            config: ObvConfig::default(),
            current_obv: dec!(0),
            previous_close: None,
            obv_history: VecDeque::new(),
            price_history: VecDeque::new(),
            volume_history: VecDeque::new(),
            timestamps: VecDeque::new(),
            is_ready: false,
        }
    }
}

/// Helper enum for trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrendDir {
    Up,
    Down,
    Sideways,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_obv_creation() {
        let config = ObvConfig::default();
        let obv = OnBalanceVolume::new(config).unwrap();
        assert_eq!(obv.current_obv, dec!(0));
        assert!(!obv.is_ready());
    }

    #[test]
    fn test_obv_calculation() {
        let mut obv = OnBalanceVolume::default();

        // Test data: price up, volume should be added
        let ohlc1 = OhlcData {
            open: dec!(100),
            high: dec!(105),
            low: dec!(99),
            close: dec!(102),
            volume: dec!(1000),
            timestamp: Utc::now(),
            quote_volume: Some(dec!(102000)),
            trade_count: Some(100),
            taker_buy_base_volume: Some(dec!(600)),
            taker_buy_quote_volume: Some(dec!(61200)),
        };

        let _result1 = obv.update(&ohlc1).unwrap();
        assert_eq!(obv.get_obv_value(), dec!(1000)); // First data point

        // Price up again
        let ohlc2 = OhlcData {
            open: dec!(102),
            high: dec!(106),
            low: dec!(101),
            close: dec!(104),
            volume: dec!(800),
            timestamp: Utc::now(),
            quote_volume: Some(dec!(83200)),
            trade_count: Some(80),
            taker_buy_base_volume: Some(dec!(480)),
            taker_buy_quote_volume: Some(dec!(49920)),
        };

        let _result2 = obv.update(&ohlc2).unwrap();
        assert_eq!(obv.get_obv_value(), dec!(1800)); // 1000 + 800

        // Price down
        let ohlc3 = OhlcData {
            open: dec!(104),
            high: dec!(105),
            low: dec!(100),
            close: dec!(101),
            volume: dec!(600),
            timestamp: Utc::now(),
            quote_volume: Some(dec!(60600)),
            trade_count: Some(60),
            taker_buy_base_volume: Some(dec!(300)),
            taker_buy_quote_volume: Some(dec!(30300)),
        };

        let _result3 = obv.update(&ohlc3).unwrap();
        assert_eq!(obv.get_obv_value(), dec!(1200)); // 1800 - 600
    }

    #[test]
    fn test_obv_trend_detection() {
        let mut obv = OnBalanceVolume::new(ObvConfig {
            signal_period: 3,
            ..Default::default()
        })
        .unwrap();

        // Add enough data points
        for i in 0..5 {
            let ohlc = OhlcData {
                open: dec!(100),
                high: dec!(105),
                low: dec!(99),
                close: dec!(100) + Decimal::from(i), // Increasing price
                volume: dec!(1000),
                timestamp: Utc::now(),
                quote_volume: Some(dec!(100000) + Decimal::from(i * 1000)),
                trade_count: Some(100),
                taker_buy_base_volume: Some(dec!(600)),
                taker_buy_quote_volume: Some(dec!(60000) + Decimal::from(i * 600)),
            };
            obv.update(&ohlc).unwrap();
        }

        assert!(obv.is_bullish());
        assert!(!obv.is_bearish());
    }
}
