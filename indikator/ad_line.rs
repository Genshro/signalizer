//! Accumulation/Distribution Line (A/D Line) Indicator
//!
//! A/D Line combines price and volume to show how money may be flowing into or out of a security.
//! It uses the Money Flow Multiplier to determine whether accumulation or distribution is occurring.
//! Rising A/D Line suggests accumulation, while falling A/D Line suggests distribution.

use crate::types::{IndicatorResult, OhlcData, Signal, SignalStrength, TechnicalAnalysisError};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};

/// A/D Line Configuration
#[derive(Debug, Clone)]
pub struct AdLineConfig {
    /// Period for signal generation (default: 14)
    pub signal_period: usize,
    /// Minimum volume threshold for calculation
    pub min_volume_threshold: Decimal,
    /// Enable divergence detection
    pub enable_divergence: bool,
}

impl Default for AdLineConfig {
    fn default() -> Self {
        Self {
            signal_period: 14,
            min_volume_threshold: dec!(0.0),
            enable_divergence: true,
        }
    }
}

/// A/D Line calculation result
#[derive(Debug, Clone)]
pub struct AdLineResult {
    /// Current A/D Line value
    pub ad_line_value: Decimal,
    /// Money Flow Multiplier
    pub money_flow_multiplier: Decimal,
    /// Money Flow Volume
    pub money_flow_volume: Decimal,
    /// A/D Line trend direction
    pub trend: AdLineTrend,
    /// Signal strength
    pub signal_strength: SignalStrength,
    /// Accumulation/Distribution status
    pub ad_status: AdStatus,
    /// Divergence detection (if enabled)
    pub divergence: Option<AdLineDivergence>,
}

/// A/D Line trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdLineTrend {
    /// Strong accumulation
    StrongAccumulation,
    /// Weak accumulation
    WeakAccumulation,
    /// Neutral/sideways
    Neutral,
    /// Weak distribution
    WeakDistribution,
    /// Strong distribution
    StrongDistribution,
}

/// Accumulation/Distribution status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdStatus {
    /// Strong accumulation
    StrongAccumulation,
    /// Moderate accumulation
    ModerateAccumulation,
    /// Neutral
    Neutral,
    /// Moderate distribution
    ModerateDistribution,
    /// Strong distribution
    StrongDistribution,
}

/// A/D Line divergence detection
#[derive(Debug, Clone)]
pub struct AdLineDivergence {
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
    /// Bullish divergence (price down, A/D Line up)
    Bullish,
    /// Bearish divergence (price up, A/D Line down)
    Bearish,
    /// No divergence
    None,
}

/// Accumulation/Distribution Line indicator
#[derive(Debug, Clone)]
pub struct AccumulationDistributionLine {
    /// Configuration
    config: AdLineConfig,
    /// Current A/D Line value
    current_ad_line: Decimal,
    /// Previous close price for trend calculation
    previous_close: Option<Decimal>,
    /// A/D Line history for trend analysis
    ad_line_history: VecDeque<Decimal>,
    /// Price history for divergence detection
    price_history: VecDeque<Decimal>,
    /// Money Flow Multiplier history
    mf_multiplier_history: VecDeque<Decimal>,
    /// Money Flow Volume history
    mf_volume_history: VecDeque<Decimal>,
    /// Timestamps for tracking
    timestamps: VecDeque<DateTime<Utc>>,
    /// Is indicator ready
    is_ready: bool,
}

impl AccumulationDistributionLine {
    /// Create new A/D Line indicator
    pub fn new(config: AdLineConfig) -> Result<Self, TechnicalAnalysisError> {
        if config.signal_period == 0 {
            return Err(TechnicalAnalysisError::invalid_config(
                "Signal period cannot be zero",
            ));
        }

        Ok(Self {
            config,
            current_ad_line: dec!(0),
            previous_close: None,
            ad_line_history: VecDeque::new(),
            price_history: VecDeque::new(),
            mf_multiplier_history: VecDeque::new(),
            mf_volume_history: VecDeque::new(),
            timestamps: VecDeque::new(),
            is_ready: false,
        })
    }

    /// Update A/D Line with new OHLC data
    pub fn update(
        &mut self,
        ohlc: &OhlcData,
    ) -> Result<Option<IndicatorResult>, TechnicalAnalysisError> {
        // Skip if volume is below threshold
        if ohlc.volume < self.config.min_volume_threshold {
            return Ok(None);
        }

        // Calculate Money Flow Multiplier
        let mf_multiplier = self.calculate_money_flow_multiplier(ohlc)?;

        // Calculate Money Flow Volume
        let mf_volume = mf_multiplier * ohlc.volume;

        // Update A/D Line
        self.current_ad_line += mf_volume;

        // Update history
        self.ad_line_history.push_back(self.current_ad_line);
        self.price_history.push_back(ohlc.close);
        self.mf_multiplier_history.push_back(mf_multiplier);
        self.mf_volume_history.push_back(mf_volume);
        self.timestamps.push_back(ohlc.timestamp);

        // Keep limited history
        let max_history = self.config.signal_period * 3;
        if self.ad_line_history.len() > max_history {
            self.ad_line_history.pop_front();
            self.price_history.pop_front();
            self.mf_multiplier_history.pop_front();
            self.mf_volume_history.pop_front();
            self.timestamps.pop_front();
        }

        // Update previous close
        self.previous_close = Some(ohlc.close);

        // Check if ready
        if self.ad_line_history.len() >= self.config.signal_period {
            self.is_ready = true;
        }

        if !self.is_ready {
            return Ok(None);
        }

        // Generate result
        let result = self.generate_result(ohlc, mf_multiplier, mf_volume)?;
        Ok(Some(result))
    }

    /// Calculate Money Flow Multiplier
    fn calculate_money_flow_multiplier(
        &self,
        ohlc: &OhlcData,
    ) -> Result<Decimal, TechnicalAnalysisError> {
        let high_low_diff = ohlc.high - ohlc.low;

        if high_low_diff == dec!(0) {
            return Ok(dec!(0)); // Doji case
        }

        let mf_multiplier = ((ohlc.close - ohlc.low) - (ohlc.high - ohlc.close)) / high_low_diff;
        Ok(mf_multiplier)
    }

    /// Generate A/D Line result
    fn generate_result(
        &self,
        ohlc: &OhlcData,
        mf_multiplier: Decimal,
        mf_volume: Decimal,
    ) -> Result<IndicatorResult, TechnicalAnalysisError> {
        let trend = self.calculate_trend();
        let signal_strength = self.calculate_signal_strength();
        let ad_status = self.calculate_ad_status(mf_multiplier);
        let divergence = if self.config.enable_divergence {
            self.detect_divergence()
        } else {
            None
        };

        let result = AdLineResult {
            ad_line_value: self.current_ad_line,
            money_flow_multiplier: mf_multiplier,
            money_flow_volume: mf_volume,
            trend,
            signal_strength,
            ad_status,
            divergence: divergence.clone(),
        };

        let signal = self.generate_signal_from_result(&result);

        let mut additional_values = HashMap::new();
        additional_values.insert("money_flow_multiplier".to_string(), mf_multiplier);
        additional_values.insert("money_flow_volume".to_string(), mf_volume);

        Ok(IndicatorResult {
            timestamp: ohlc.timestamp,
            value: self.current_ad_line,
            signal: Some(signal),
            confidence: Some(self.calculate_confidence(&result)),
            indicator_name: "A/D Line".to_string(),
            additional_values: Some(additional_values),
            metadata: Some(serde_json::json!({
                "trend": format!("{:?}", trend),
                "signal_strength": format!("{:?}", signal_strength),
                "ad_status": format!("{:?}", ad_status),
                "divergence": divergence.as_ref().map(|d| format!("{:?}", d.divergence_type))
            })),
        })
    }

    /// Calculate A/D Line trend
    fn calculate_trend(&self) -> AdLineTrend {
        if self.ad_line_history.len() < self.config.signal_period {
            return AdLineTrend::Neutral;
        }

        let recent_values: Vec<Decimal> = self
            .ad_line_history
            .iter()
            .rev()
            .take(self.config.signal_period)
            .cloned()
            .collect();

        let trend_direction = self.calculate_trend_direction(&recent_values);
        let trend_strength = self.calculate_trend_strength(&recent_values);

        match trend_direction {
            TrendDir::Up => {
                if trend_strength > dec!(0.7) {
                    AdLineTrend::StrongAccumulation
                } else {
                    AdLineTrend::WeakAccumulation
                }
            },
            TrendDir::Down => {
                if trend_strength > dec!(0.7) {
                    AdLineTrend::StrongDistribution
                } else {
                    AdLineTrend::WeakDistribution
                }
            },
            TrendDir::Sideways => AdLineTrend::Neutral,
        }
    }

    /// Calculate Accumulation/Distribution status
    fn calculate_ad_status(&self, mf_multiplier: Decimal) -> AdStatus {
        if mf_multiplier > dec!(0.5) {
            AdStatus::StrongAccumulation
        } else if mf_multiplier > dec!(0.2) {
            AdStatus::ModerateAccumulation
        } else if mf_multiplier < dec!(-0.5) {
            AdStatus::StrongDistribution
        } else if mf_multiplier < dec!(-0.2) {
            AdStatus::ModerateDistribution
        } else {
            AdStatus::Neutral
        }
    }

    /// Calculate signal strength
    fn calculate_signal_strength(&self) -> SignalStrength {
        if self.ad_line_history.len() < self.config.signal_period {
            return SignalStrength::Weak;
        }

        let recent_values: Vec<Decimal> = self
            .ad_line_history
            .iter()
            .rev()
            .take(self.config.signal_period)
            .cloned()
            .collect();

        let trend_strength = self.calculate_trend_strength(&recent_values);

        if trend_strength > dec!(0.8) {
            SignalStrength::Strong
        } else if trend_strength > dec!(0.5) {
            SignalStrength::Moderate
        } else {
            SignalStrength::Weak
        }
    }

    /// Detect divergence
    fn detect_divergence(&self) -> Option<AdLineDivergence> {
        if !self.config.enable_divergence
            || self.ad_line_history.len() < self.config.signal_period * 2
            || self.price_history.len() < self.config.signal_period * 2
        {
            return None;
        }

        let period = self.config.signal_period;

        let recent_ad: Vec<Decimal> = self
            .ad_line_history
            .iter()
            .rev()
            .take(period)
            .cloned()
            .collect();

        let previous_ad: Vec<Decimal> = self
            .ad_line_history
            .iter()
            .rev()
            .skip(period)
            .take(period)
            .cloned()
            .collect();

        let recent_prices: Vec<Decimal> = self
            .price_history
            .iter()
            .rev()
            .take(period)
            .cloned()
            .collect();

        let previous_prices: Vec<Decimal> = self
            .price_history
            .iter()
            .rev()
            .skip(period)
            .take(period)
            .cloned()
            .collect();

        if recent_ad.is_empty()
            || previous_ad.is_empty()
            || recent_prices.is_empty()
            || previous_prices.is_empty()
        {
            return None;
        }

        let ad_direction = self.calculate_trend_direction(&recent_ad);
        let price_direction = self.calculate_trend_direction(&recent_prices);

        let divergence_type = match (ad_direction, price_direction) {
            (TrendDir::Up, TrendDir::Down) => DivergenceType::Bullish,
            (TrendDir::Down, TrendDir::Up) => DivergenceType::Bearish,
            _ => DivergenceType::None,
        };

        if divergence_type != DivergenceType::None {
            let strength = self.calculate_divergence_strength(&recent_prices, &recent_ad);
            Some(AdLineDivergence {
                divergence_type,
                strength,
                periods: period,
            })
        } else {
            None
        }
    }

    /// Calculate trend direction
    fn calculate_trend_direction(&self, values: &[Decimal]) -> TrendDir {
        if values.len() < 2 {
            return TrendDir::Sideways;
        }

        let start = values[values.len() - 1];
        let end = values[0];
        let change_ratio = if start != dec!(0) {
            (end - start) / start.abs()
        } else {
            dec!(0)
        };

        if change_ratio > dec!(0.01) {
            TrendDir::Up
        } else if change_ratio < dec!(-0.01) {
            TrendDir::Down
        } else {
            TrendDir::Sideways
        }
    }

    /// Calculate trend strength
    fn calculate_trend_strength(&self, values: &[Decimal]) -> Decimal {
        if values.len() < 2 {
            return dec!(0);
        }

        let start = values[values.len() - 1];
        let end = values[0];

        if start == dec!(0) {
            return dec!(0);
        }

        let change_ratio = ((end - start) / start.abs()).abs();
        change_ratio.min(dec!(1))
    }

    /// Calculate divergence strength
    fn calculate_divergence_strength(&self, prices: &[Decimal], ad_values: &[Decimal]) -> Decimal {
        if prices.len() < 2 || ad_values.len() < 2 {
            return dec!(0);
        }

        let price_strength = self.calculate_trend_strength(prices);
        let ad_strength = self.calculate_trend_strength(ad_values);

        (price_strength + ad_strength) / dec!(2)
    }

    /// Generate signal from result
    fn generate_signal_from_result(&self, result: &AdLineResult) -> Signal {
        match result.trend {
            AdLineTrend::StrongAccumulation => Signal::Buy,
            AdLineTrend::StrongDistribution => Signal::Sell,
            _ => {
                if let Some(divergence) = &result.divergence {
                    match divergence.divergence_type {
                        DivergenceType::Bullish => Signal::Buy,
                        DivergenceType::Bearish => Signal::Sell,
                        _ => Signal::Neutral,
                    }
                } else {
                    Signal::Neutral
                }
            },
        }
    }

    /// Calculate confidence level
    fn calculate_confidence(&self, result: &AdLineResult) -> Decimal {
        let mut confidence = dec!(0.5); // Base confidence

        // Adjust based on trend strength
        match result.trend {
            AdLineTrend::StrongAccumulation | AdLineTrend::StrongDistribution => {
                confidence += dec!(0.3);
            },
            AdLineTrend::WeakAccumulation | AdLineTrend::WeakDistribution => {
                confidence += dec!(0.1);
            },
            _ => {},
        }

        // Adjust based on divergence
        if let Some(divergence) = &result.divergence {
            if divergence.divergence_type != DivergenceType::None {
                confidence += divergence.strength * dec!(0.2);
            }
        }

        confidence.min(dec!(1)).max(dec!(0))
    }

    /// Generate signal
    pub fn generate_signal(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<crate::types::SignalData>, TechnicalAnalysisError> {
        if !self.is_ready {
            return Ok(None);
        }

        // Get latest result data
        let trend = self.calculate_trend();
        let signal_strength = self.calculate_signal_strength();

        let signal = match trend {
            AdLineTrend::StrongAccumulation => Signal::Buy,
            AdLineTrend::StrongDistribution => Signal::Sell,
            _ => Signal::Neutral,
        };

        let confidence = match signal_strength {
            SignalStrength::VeryStrong => dec!(0.95),
            SignalStrength::Strong => dec!(0.9),
            SignalStrength::Moderate => dec!(0.7),
            SignalStrength::Weak => dec!(0.5),
            SignalStrength::VeryWeak => dec!(0.3),
        };

        Ok(Some(crate::types::SignalData {
            signal,
            strength: signal_strength,
            confidence,
            timestamp,
            price: Some(self.price_history.back().copied().unwrap_or(dec!(0))),
            volume: Some(dec!(0)), // Will be filled by caller
            timeframe: Some(crate::types::Timeframe::M1), // Default, will be set by caller
            metadata: Some(serde_json::json!({
                "ad_line_value": self.current_ad_line,
                "trend": format!("{:?}", trend),
                "signal_strength": format!("{:?}", signal_strength)
            })),
            duration: None,
            source: "A/D Line".to_string(),
            stop_loss: None,
            take_profit: None,
            risk_reward_ratio: None,
        }))
    }

    /// Check if A/D Line is bullish
    pub fn is_bullish(&self) -> bool {
        matches!(
            self.calculate_trend(),
            AdLineTrend::StrongAccumulation | AdLineTrend::WeakAccumulation
        )
    }

    /// Check if A/D Line is bearish
    pub fn is_bearish(&self) -> bool {
        matches!(
            self.calculate_trend(),
            AdLineTrend::StrongDistribution | AdLineTrend::WeakDistribution
        )
    }

    /// Get current A/D Line value
    pub fn get_ad_line_value(&self) -> Decimal {
        self.current_ad_line
    }

    /// Check if indicator is ready
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    /// Reset indicator
    pub fn reset(&mut self) {
        self.current_ad_line = dec!(0);
        self.previous_close = None;
        self.ad_line_history.clear();
        self.price_history.clear();
        self.mf_multiplier_history.clear();
        self.mf_volume_history.clear();
        self.timestamps.clear();
        self.is_ready = false;
    }
}

impl Default for AccumulationDistributionLine {
    fn default() -> Self {
        Self::new(AdLineConfig::default()).expect("Default A/D Line configuration should be valid")
    }
}

/// Trend direction helper enum
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
    fn test_ad_line_creation() {
        let config = AdLineConfig::default();
        let ad_line = AccumulationDistributionLine::new(config);
        assert!(ad_line.is_ok());
    }

    #[test]
    fn test_ad_line_calculation() {
        let config = AdLineConfig::default();
        let mut ad_line = AccumulationDistributionLine::new(config).unwrap();

        let ohlc = OhlcData {
            timestamp: Utc::now(),
            open: dec!(100),
            high: dec!(105),
            low: dec!(95),
            close: dec!(102),
            volume: dec!(1000),
            quote_volume: Some(dec!(100000)),
            trade_count: Some(100),
            taker_buy_base_volume: Some(dec!(500)),
            taker_buy_quote_volume: Some(dec!(50000)),
        };

        let result = ad_line.update(&ohlc);
        assert!(result.is_ok());
    }

    #[test]
    fn test_money_flow_multiplier() {
        let config = AdLineConfig::default();
        let ad_line = AccumulationDistributionLine::new(config).unwrap();

        let ohlc = OhlcData {
            timestamp: Utc::now(),
            open: dec!(100),
            high: dec!(105),
            low: dec!(95),
            close: dec!(102),
            volume: dec!(1000),
            quote_volume: Some(dec!(100000)),
            trade_count: Some(100),
            taker_buy_base_volume: Some(dec!(500)),
            taker_buy_quote_volume: Some(dec!(50000)),
        };

        let mf_multiplier = ad_line.calculate_money_flow_multiplier(&ohlc).unwrap();
        // ((102 - 95) - (105 - 102)) / (105 - 95) = (7 - 3) / 10 = 0.4
        assert_eq!(mf_multiplier, dec!(0.4));
    }

    #[test]
    fn test_ad_line_accumulation() {
        let config = AdLineConfig::default();
        let mut ad_line = AccumulationDistributionLine::new(config).unwrap();

        // Test accumulation pattern (closes near highs)
        let test_data = vec![
            OhlcData {
                timestamp: Utc::now(),
                open: dec!(100),
                high: dec!(105),
                low: dec!(98),
                close: dec!(104), // Close near high
                volume: dec!(1000),
                quote_volume: Some(dec!(100000)),
                trade_count: Some(100),
                taker_buy_base_volume: Some(dec!(500)),
                taker_buy_quote_volume: Some(dec!(50000)),
            },
            OhlcData {
                timestamp: Utc::now(),
                open: dec!(104),
                high: dec!(109),
                low: dec!(102),
                close: dec!(108), // Close near high
                volume: dec!(1200),
                quote_volume: Some(dec!(120000)),
                trade_count: Some(120),
                taker_buy_base_volume: Some(dec!(600)),
                taker_buy_quote_volume: Some(dec!(60000)),
            },
        ];

        for ohlc in &test_data {
            let _ = ad_line.update(ohlc);
        }

        // A/D Line should be positive due to accumulation pattern
        assert!(ad_line.get_ad_line_value() > dec!(0));
    }

    #[test]
    fn test_ad_line_distribution() {
        let config = AdLineConfig::default();
        let mut ad_line = AccumulationDistributionLine::new(config).unwrap();

        // Test distribution pattern (closes near lows)
        let test_data = vec![
            OhlcData {
                timestamp: Utc::now(),
                open: dec!(100),
                high: dec!(105),
                low: dec!(98),
                close: dec!(99), // Close near low
                volume: dec!(1000),
                quote_volume: Some(dec!(100000)),
                trade_count: Some(100),
                taker_buy_base_volume: Some(dec!(500)),
                taker_buy_quote_volume: Some(dec!(50000)),
            },
            OhlcData {
                timestamp: Utc::now(),
                open: dec!(99),
                high: dec!(104),
                low: dec!(96),
                close: dec!(97), // Close near low
                volume: dec!(1200),
                quote_volume: Some(dec!(120000)),
                trade_count: Some(120),
                taker_buy_base_volume: Some(dec!(600)),
                taker_buy_quote_volume: Some(dec!(60000)),
            },
        ];

        for ohlc in &test_data {
            let _ = ad_line.update(ohlc);
        }

        // A/D Line should be negative due to distribution pattern
        assert!(ad_line.get_ad_line_value() < dec!(0));
    }
}
