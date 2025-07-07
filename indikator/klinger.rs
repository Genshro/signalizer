//! Klinger Oscillator Indicator
//!
//! The Klinger Oscillator is a volume-based momentum oscillator that combines price and volume
//! to identify long-term money flow while remaining sensitive to short-term fluctuations.
//! It uses two exponential moving averages of the Volume Force (VF) to generate signals.
//!
//! Volume Force = Volume × Trend × Direction
//! Where:
//! - Trend = DM (Daily Movement) / CM (Cumulative Movement)
//! - Direction = +1 if (H+L+C) > Previous (H+L+C), else -1
//! - DM = High - Low
//! - CM = Sum of DM over periods where Direction doesn't change

use crate::types::{IndicatorResult, OhlcData, Signal, SignalStrength, TechnicalAnalysisError};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};

/// Klinger Oscillator Configuration
#[derive(Debug, Clone)]
pub struct KlingerConfig {
    /// Fast EMA period (default: 34)
    pub fast_period: usize,
    /// Slow EMA period (default: 55)
    pub slow_period: usize,
    /// Signal line EMA period (default: 13)
    pub signal_period: usize,
    /// Minimum volume threshold for calculation
    pub min_volume_threshold: Decimal,
    /// Enable divergence detection
    pub enable_divergence: bool,
    /// Overbought threshold (positive value)
    pub overbought_threshold: Decimal,
    /// Oversold threshold (negative value)
    pub oversold_threshold: Decimal,
}

impl Default for KlingerConfig {
    fn default() -> Self {
        Self {
            fast_period: 34,
            slow_period: 55,
            signal_period: 13,
            min_volume_threshold: dec!(0.0),
            enable_divergence: true,
            overbought_threshold: dec!(1000),
            oversold_threshold: dec!(-1000),
        }
    }
}

/// Klinger Oscillator calculation result
#[derive(Debug, Clone)]
pub struct KlingerResult {
    /// Klinger Oscillator value (Fast EMA - Slow EMA)
    pub klinger_value: Decimal,
    /// Signal line value (EMA of Klinger)
    pub signal_line: Decimal,
    /// Histogram (Klinger - Signal)
    pub histogram: Decimal,
    /// Volume Force value
    pub volume_force: Decimal,
    /// Trend direction
    pub trend: KlingerTrend,
    /// Signal strength
    pub signal_strength: SignalStrength,
    /// Money flow direction
    pub money_flow: MoneyFlowDirection,
    /// Divergence detection (if enabled)
    pub divergence: Option<KlingerDivergence>,
}

/// Klinger trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KlingerTrend {
    /// Strong bullish trend
    StrongBullish,
    /// Weak bullish trend
    WeakBullish,
    /// Neutral trend
    Neutral,
    /// Weak bearish trend
    WeakBearish,
    /// Strong bearish trend
    StrongBearish,
}

/// Money flow direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoneyFlowDirection {
    /// Strong inflow
    StrongInflow,
    /// Moderate inflow
    ModerateInflow,
    /// Neutral flow
    Neutral,
    /// Moderate outflow
    ModerateOutflow,
    /// Strong outflow
    StrongOutflow,
}

/// Klinger divergence detection
#[derive(Debug, Clone)]
pub struct KlingerDivergence {
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
    /// Bullish divergence (price down, Klinger up)
    Bullish,
    /// Bearish divergence (price up, Klinger down)
    Bearish,
    /// No divergence
    None,
}

/// Internal trend tracking for Volume Force calculation
#[derive(Debug, Clone)]
struct TrendState {
    /// Current direction (+1 or -1)
    direction: i8,
    /// Cumulative movement in current direction
    cumulative_movement: Decimal,
    /// Previous typical price
    previous_typical_price: Option<Decimal>,
}

/// Klinger Oscillator indicator
#[derive(Debug, Clone)]
pub struct KlingerOscillator {
    /// Configuration
    config: KlingerConfig,
    /// OHLC data history
    data_history: VecDeque<OhlcData>,
    /// Volume Force history
    volume_force_history: VecDeque<Decimal>,
    /// Fast EMA values
    fast_ema_values: VecDeque<Decimal>,
    /// Slow EMA values
    slow_ema_values: VecDeque<Decimal>,
    /// Klinger oscillator values
    klinger_values: VecDeque<Decimal>,
    /// Signal line values
    signal_values: VecDeque<Decimal>,
    /// Price history for divergence detection
    price_history: VecDeque<Decimal>,
    /// Trend state for Volume Force calculation
    trend_state: TrendState,
    /// Current fast EMA
    current_fast_ema: Option<Decimal>,
    /// Current slow EMA
    current_slow_ema: Option<Decimal>,
    /// Current signal line
    current_signal: Option<Decimal>,
    /// Timestamps for tracking
    timestamps: VecDeque<DateTime<Utc>>,
    /// Is indicator ready
    is_ready: bool,
}

impl KlingerOscillator {
    /// Create new Klinger Oscillator indicator
    pub fn new(config: KlingerConfig) -> Result<Self, TechnicalAnalysisError> {
        if config.fast_period == 0 || config.slow_period == 0 || config.signal_period == 0 {
            return Err(TechnicalAnalysisError::invalid_config(
                "Periods cannot be zero",
            ));
        }

        if config.fast_period >= config.slow_period {
            return Err(TechnicalAnalysisError::invalid_config(
                "Fast period must be less than slow period",
            ));
        }

        Ok(Self {
            config,
            data_history: VecDeque::new(),
            volume_force_history: VecDeque::new(),
            fast_ema_values: VecDeque::new(),
            slow_ema_values: VecDeque::new(),
            klinger_values: VecDeque::new(),
            signal_values: VecDeque::new(),
            price_history: VecDeque::new(),
            trend_state: TrendState {
                direction: 1,
                cumulative_movement: dec!(0),
                previous_typical_price: None,
            },
            current_fast_ema: None,
            current_slow_ema: None,
            current_signal: None,
            timestamps: VecDeque::new(),
            is_ready: false,
        })
    }

    /// Update Klinger Oscillator with new OHLC data
    pub fn update(
        &mut self,
        ohlc: &OhlcData,
    ) -> Result<Option<IndicatorResult>, TechnicalAnalysisError> {
        // Skip if volume is below threshold
        if ohlc.volume < self.config.min_volume_threshold {
            return Ok(None);
        }

        // Update data history
        self.data_history.push_back(ohlc.clone());
        self.price_history.push_back(ohlc.close);
        self.timestamps.push_back(ohlc.timestamp);

        // Calculate Volume Force
        let volume_force = self.calculate_volume_force(ohlc)?;
        self.volume_force_history.push_back(volume_force);

        // Calculate EMAs - use separate method calls to avoid borrow conflicts
        let fast_ema = {
            let multiplier = dec!(2) / (Decimal::from(self.config.fast_period) + dec!(1));
            let ema = if let Some(&previous_ema) = self.fast_ema_values.back() {
                volume_force * multiplier + previous_ema * (dec!(1) - multiplier)
            } else {
                volume_force
            };
            self.fast_ema_values.push_back(ema);
            if self.fast_ema_values.len() > self.config.fast_period * 2 {
                self.fast_ema_values.pop_front();
            }
            ema
        };

        let slow_ema = {
            let multiplier = dec!(2) / (Decimal::from(self.config.slow_period) + dec!(1));
            let ema = if let Some(&previous_ema) = self.slow_ema_values.back() {
                volume_force * multiplier + previous_ema * (dec!(1) - multiplier)
            } else {
                volume_force
            };
            self.slow_ema_values.push_back(ema);
            if self.slow_ema_values.len() > self.config.slow_period * 2 {
                self.slow_ema_values.pop_front();
            }
            ema
        };

        self.current_fast_ema = Some(fast_ema);
        self.current_slow_ema = Some(slow_ema);

        // Calculate Klinger Oscillator (Fast EMA - Slow EMA)
        let klinger_value = fast_ema - slow_ema;
        self.klinger_values.push_back(klinger_value);

        // Calculate Signal Line (EMA of Klinger)
        let signal_line = {
            let multiplier = dec!(2) / (Decimal::from(self.config.signal_period) + dec!(1));
            let ema = if let Some(&previous_ema) = self.signal_values.back() {
                klinger_value * multiplier + previous_ema * (dec!(1) - multiplier)
            } else {
                klinger_value
            };
            self.signal_values.push_back(ema);
            if self.signal_values.len() > self.config.signal_period * 2 {
                self.signal_values.pop_front();
            }
            ema
        };

        self.current_signal = Some(signal_line);

        // Keep limited history
        let max_history = self.config.slow_period * 3;
        self.maintain_history_limits(max_history);

        // Check if ready
        if self.klinger_values.len() >= self.config.signal_period {
            self.is_ready = true;
        }

        if !self.is_ready {
            return Ok(None);
        }

        // Generate result
        let result = self.generate_result(ohlc, klinger_value, signal_line, volume_force)?;
        Ok(Some(result))
    }

    /// Calculate Volume Force for current period
    fn calculate_volume_force(
        &mut self,
        ohlc: &OhlcData,
    ) -> Result<Decimal, TechnicalAnalysisError> {
        // Calculate typical price
        let typical_price = (ohlc.high + ohlc.low + ohlc.close) / dec!(3);

        // Calculate daily movement (DM)
        let daily_movement = ohlc.high - ohlc.low;

        // Determine direction
        let direction = if let Some(prev_typical) = self.trend_state.previous_typical_price {
            if typical_price > prev_typical {
                1
            } else if typical_price < prev_typical {
                -1
            } else {
                self.trend_state.direction // Keep previous direction if no change
            }
        } else {
            1 // Default to positive for first calculation
        };

        // Update cumulative movement
        if direction == self.trend_state.direction {
            // Same direction, add to cumulative movement
            self.trend_state.cumulative_movement += daily_movement;
        } else {
            // Direction changed, reset cumulative movement
            self.trend_state.cumulative_movement = daily_movement;
            self.trend_state.direction = direction;
        }

        // Calculate trend (DM / CM)
        let trend = if self.trend_state.cumulative_movement != dec!(0) {
            daily_movement / self.trend_state.cumulative_movement
        } else {
            dec!(1) // Default to 1 if no cumulative movement
        };

        // Calculate Volume Force
        let volume_force = ohlc.volume * trend * Decimal::from(direction);

        // Update previous typical price
        self.trend_state.previous_typical_price = Some(typical_price);

        Ok(volume_force)
    }

    /// Calculate EMA for given value and period
    fn calculate_ema(
        &self,
        value: Decimal,
        period: usize,
        ema_values: &mut VecDeque<Decimal>,
    ) -> Result<Decimal, TechnicalAnalysisError> {
        let multiplier = dec!(2) / (Decimal::from(period) + dec!(1));

        let ema = if let Some(&previous_ema) = ema_values.back() {
            // EMA = (Value * Multiplier) + (Previous EMA * (1 - Multiplier))
            value * multiplier + previous_ema * (dec!(1) - multiplier)
        } else {
            // First EMA value is the value itself
            value
        };

        ema_values.push_back(ema);

        // Keep limited EMA history
        if ema_values.len() > period * 2 {
            ema_values.pop_front();
        }

        Ok(ema)
    }

    /// Generate indicator result
    fn generate_result(
        &self,
        ohlc: &OhlcData,
        klinger_value: Decimal,
        signal_line: Decimal,
        volume_force: Decimal,
    ) -> Result<IndicatorResult, TechnicalAnalysisError> {
        let histogram = klinger_value - signal_line;
        let trend = self.calculate_trend(klinger_value, signal_line);
        let signal_strength = self.calculate_signal_strength(klinger_value, histogram);
        let money_flow = self.calculate_money_flow(volume_force, klinger_value);
        let divergence = if self.config.enable_divergence {
            self.detect_divergence()
        } else {
            None
        };

        let result = KlingerResult {
            klinger_value,
            signal_line,
            histogram,
            volume_force,
            trend,
            signal_strength,
            money_flow,
            divergence: divergence.clone(),
        };

        let signal = self.generate_signal_from_result(&result);

        let mut additional_values = HashMap::new();
        additional_values.insert("signal_line".to_string(), signal_line);
        additional_values.insert("histogram".to_string(), histogram);
        additional_values.insert("volume_force".to_string(), volume_force);

        Ok(IndicatorResult {
            timestamp: ohlc.timestamp,
            value: klinger_value,
            signal: Some(signal),
            confidence: Some(self.calculate_confidence(&result)),
            indicator_name: "Klinger Oscillator".to_string(),
            additional_values: Some(additional_values),
            metadata: Some(serde_json::json!({
                "trend": format!("{:?}", trend),
                "signal_strength": format!("{:?}", signal_strength),
                "money_flow": format!("{:?}", money_flow),
                "divergence": divergence.as_ref().map(|d| format!("{:?}", d.divergence_type))
            })),
        })
    }

    /// Calculate trend based on Klinger and signal line
    fn calculate_trend(&self, klinger_value: Decimal, signal_line: Decimal) -> KlingerTrend {
        let diff = klinger_value - signal_line;

        // Use thresholds based on recent volatility
        let strong_threshold = self.config.overbought_threshold / dec!(2);
        let weak_threshold = strong_threshold / dec!(2);

        if diff > strong_threshold {
            KlingerTrend::StrongBullish
        } else if diff > weak_threshold {
            KlingerTrend::WeakBullish
        } else if diff < -strong_threshold {
            KlingerTrend::StrongBearish
        } else if diff < -weak_threshold {
            KlingerTrend::WeakBearish
        } else {
            KlingerTrend::Neutral
        }
    }

    /// Calculate signal strength
    fn calculate_signal_strength(
        &self,
        klinger_value: Decimal,
        histogram: Decimal,
    ) -> SignalStrength {
        let klinger_abs = klinger_value.abs();
        let histogram_abs = histogram.abs();

        // Strong signals when both Klinger and histogram are significant
        if klinger_abs > self.config.overbought_threshold
            && histogram_abs > self.config.overbought_threshold / dec!(4)
        {
            SignalStrength::VeryStrong
        } else if klinger_abs > self.config.overbought_threshold / dec!(2)
            && histogram_abs > self.config.overbought_threshold / dec!(8)
        {
            SignalStrength::Strong
        } else if klinger_abs > self.config.overbought_threshold / dec!(4) {
            SignalStrength::Moderate
        } else if klinger_abs > self.config.overbought_threshold / dec!(8) {
            SignalStrength::Weak
        } else {
            SignalStrength::VeryWeak
        }
    }

    /// Calculate money flow direction
    fn calculate_money_flow(
        &self,
        volume_force: Decimal,
        klinger_value: Decimal,
    ) -> MoneyFlowDirection {
        let vf_threshold = volume_force.abs() / dec!(2);
        let klinger_threshold = self.config.overbought_threshold / dec!(4);

        if volume_force > vf_threshold && klinger_value > klinger_threshold {
            MoneyFlowDirection::StrongInflow
        } else if volume_force > dec!(0) && klinger_value > dec!(0) {
            MoneyFlowDirection::ModerateInflow
        } else if volume_force < -vf_threshold && klinger_value < -klinger_threshold {
            MoneyFlowDirection::StrongOutflow
        } else if volume_force < dec!(0) && klinger_value < dec!(0) {
            MoneyFlowDirection::ModerateOutflow
        } else {
            MoneyFlowDirection::Neutral
        }
    }

    /// Detect divergence between price and Klinger
    fn detect_divergence(&self) -> Option<KlingerDivergence> {
        if self.klinger_values.len() < 20 || self.price_history.len() < 20 {
            return None;
        }

        let period = 10;
        let recent_klinger: Vec<Decimal> = self
            .klinger_values
            .iter()
            .rev()
            .take(period)
            .cloned()
            .collect();
        let previous_klinger: Vec<Decimal> = self
            .klinger_values
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

        if recent_klinger.is_empty()
            || previous_klinger.is_empty()
            || recent_prices.is_empty()
            || previous_prices.is_empty()
        {
            return None;
        }

        // Calculate trends
        let klinger_trend_recent = recent_klinger[0] - recent_klinger[period - 1];
        let klinger_trend_previous = previous_klinger[0] - previous_klinger[period - 1];
        let price_trend_recent = recent_prices[0] - recent_prices[period - 1];
        let price_trend_previous = previous_prices[0] - previous_prices[period - 1];

        // Check for divergence
        let divergence_type = if price_trend_recent < dec!(0)
            && klinger_trend_recent > dec!(0)
            && price_trend_previous < dec!(0)
            && klinger_trend_previous > dec!(0)
        {
            DivergenceType::Bullish
        } else if price_trend_recent > dec!(0)
            && klinger_trend_recent < dec!(0)
            && price_trend_previous > dec!(0)
            && klinger_trend_previous < dec!(0)
        {
            DivergenceType::Bearish
        } else {
            DivergenceType::None
        };

        if divergence_type != DivergenceType::None {
            let strength = self.calculate_divergence_strength(&recent_prices, &recent_klinger);
            Some(KlingerDivergence {
                divergence_type,
                strength,
                periods: period,
            })
        } else {
            None
        }
    }

    /// Calculate divergence strength
    fn calculate_divergence_strength(
        &self,
        prices: &[Decimal],
        klinger_values: &[Decimal],
    ) -> Decimal {
        if prices.len() < 2 || klinger_values.len() < 2 {
            return dec!(0);
        }

        let price_change = (prices[0] - prices[prices.len() - 1]).abs();
        let klinger_change = (klinger_values[0] - klinger_values[klinger_values.len() - 1]).abs();

        // Normalize and combine changes
        let price_strength = price_change / prices[0].abs().max(dec!(1));
        let klinger_strength = klinger_change / klinger_values[0].abs().max(dec!(1));

        (price_strength + klinger_strength) / dec!(2)
    }

    /// Generate signal from result
    fn generate_signal_from_result(&self, result: &KlingerResult) -> Signal {
        match result.trend {
            KlingerTrend::StrongBullish => {
                if result.histogram > dec!(0) {
                    Signal::StrongBuy
                } else {
                    Signal::Buy
                }
            },
            KlingerTrend::WeakBullish => Signal::Buy,
            KlingerTrend::StrongBearish => {
                if result.histogram < dec!(0) {
                    Signal::StrongSell
                } else {
                    Signal::Sell
                }
            },
            KlingerTrend::WeakBearish => Signal::Sell,
            KlingerTrend::Neutral => {
                // Check for crossovers
                if result.histogram > dec!(0) && result.klinger_value > result.signal_line {
                    Signal::Buy
                } else if result.histogram < dec!(0) && result.klinger_value < result.signal_line {
                    Signal::Sell
                } else {
                    Signal::Neutral
                }
            },
        }
    }

    /// Calculate confidence level
    fn calculate_confidence(&self, result: &KlingerResult) -> Decimal {
        let mut confidence = dec!(0.5); // Base confidence

        // Adjust based on signal strength
        match result.signal_strength {
            SignalStrength::VeryStrong => confidence += dec!(0.4),
            SignalStrength::Strong => confidence += dec!(0.3),
            SignalStrength::Moderate => confidence += dec!(0.2),
            SignalStrength::Weak => confidence += dec!(0.1),
            SignalStrength::VeryWeak => confidence -= dec!(0.1),
        }

        // Adjust based on trend consistency
        match result.trend {
            KlingerTrend::StrongBullish | KlingerTrend::StrongBearish => confidence += dec!(0.2),
            KlingerTrend::WeakBullish | KlingerTrend::WeakBearish => confidence += dec!(0.1),
            KlingerTrend::Neutral => confidence -= dec!(0.1),
        }

        // Adjust based on histogram alignment
        if (result.klinger_value > dec!(0) && result.histogram > dec!(0))
            || (result.klinger_value < dec!(0) && result.histogram < dec!(0))
        {
            confidence += dec!(0.1);
        }

        // Adjust based on divergence
        if let Some(ref divergence) = result.divergence {
            confidence += divergence.strength * dec!(0.2);
        }

        confidence.min(dec!(1)).max(dec!(0))
    }

    /// Maintain history limits
    fn maintain_history_limits(&mut self, max_history: usize) {
        if self.data_history.len() > max_history {
            self.data_history.pop_front();
        }
        if self.volume_force_history.len() > max_history {
            self.volume_force_history.pop_front();
        }
        if self.klinger_values.len() > max_history {
            self.klinger_values.pop_front();
        }
        if self.price_history.len() > max_history {
            self.price_history.pop_front();
        }
        if self.timestamps.len() > max_history {
            self.timestamps.pop_front();
        }
    }

    /// Generate signal
    pub fn generate_signal(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<crate::types::SignalData>, TechnicalAnalysisError> {
        if !self.is_ready {
            return Ok(None);
        }

        let klinger_value = self.klinger_values.back().copied().unwrap_or(dec!(0));
        let signal_line = self.current_signal.unwrap_or(dec!(0));
        let volume_force = self.volume_force_history.back().copied().unwrap_or(dec!(0));

        let result = KlingerResult {
            klinger_value,
            signal_line,
            histogram: klinger_value - signal_line,
            volume_force,
            trend: self.calculate_trend(klinger_value, signal_line),
            signal_strength: self
                .calculate_signal_strength(klinger_value, klinger_value - signal_line),
            money_flow: self.calculate_money_flow(volume_force, klinger_value),
            divergence: if self.config.enable_divergence {
                self.detect_divergence()
            } else {
                None
            },
        };

        let signal = self.generate_signal_from_result(&result);
        let confidence = self.calculate_confidence(&result);

        Ok(Some(crate::types::SignalData {
            signal,
            strength: result.signal_strength,
            confidence,
            timestamp,
            price: Some(self.price_history.back().copied().unwrap_or(dec!(0))),
            volume: Some(
                self.data_history
                    .back()
                    .map(|d| d.volume)
                    .unwrap_or(dec!(0)),
            ),
            timeframe: Some(crate::types::Timeframe::M1), // Default, will be set by caller
            metadata: Some(serde_json::json!({
                "klinger_value": klinger_value,
                "signal_line": signal_line,
                "histogram": result.histogram,
                "volume_force": volume_force,
                "trend": format!("{:?}", result.trend),
                "money_flow": format!("{:?}", result.money_flow)
            })),
            duration: None,
            source: "Klinger Oscillator".to_string(),
            stop_loss: None,
            take_profit: None,
            risk_reward_ratio: None,
        }))
    }

    /// Check if Klinger is bullish
    pub fn is_bullish(&self) -> bool {
        if let (Some(klinger), Some(signal)) = (self.klinger_values.back(), self.current_signal) {
            *klinger > signal && *klinger > dec!(0)
        } else {
            false
        }
    }

    /// Check if Klinger is bearish
    pub fn is_bearish(&self) -> bool {
        if let (Some(klinger), Some(signal)) = (self.klinger_values.back(), self.current_signal) {
            *klinger < signal && *klinger < dec!(0)
        } else {
            false
        }
    }

    /// Get current Klinger value
    pub fn get_klinger_value(&self) -> Option<Decimal> {
        self.klinger_values.back().copied()
    }

    /// Get current signal line value
    pub fn get_signal_line(&self) -> Option<Decimal> {
        self.current_signal
    }

    /// Get current histogram value
    pub fn get_histogram(&self) -> Option<Decimal> {
        if let (Some(klinger), Some(signal)) = (self.klinger_values.back(), self.current_signal) {
            Some(*klinger - signal)
        } else {
            None
        }
    }

    /// Check if indicator is ready
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    /// Reset indicator
    pub fn reset(&mut self) {
        self.data_history.clear();
        self.volume_force_history.clear();
        self.fast_ema_values.clear();
        self.slow_ema_values.clear();
        self.klinger_values.clear();
        self.signal_values.clear();
        self.price_history.clear();
        self.trend_state = TrendState {
            direction: 1,
            cumulative_movement: dec!(0),
            previous_typical_price: None,
        };
        self.current_fast_ema = None;
        self.current_slow_ema = None;
        self.current_signal = None;
        self.timestamps.clear();
        self.is_ready = false;
    }
}

impl Default for KlingerOscillator {
    fn default() -> Self {
        Self::new(KlingerConfig::default()).expect("Default Klinger configuration should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_klinger_creation() {
        let config = KlingerConfig::default();
        let klinger = KlingerOscillator::new(config);
        assert!(klinger.is_ok());
    }

    #[test]
    fn test_klinger_calculation() {
        let config = KlingerConfig {
            fast_period: 5,
            slow_period: 10,
            signal_period: 3,
            ..Default::default()
        };
        let mut klinger = KlingerOscillator::new(config).unwrap();

        let test_data = vec![
            OhlcData {
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
            },
            OhlcData {
                timestamp: Utc::now(),
                open: dec!(102),
                high: dec!(107),
                low: dec!(100),
                close: dec!(105),
                volume: dec!(1500),
                quote_volume: Some(dec!(150000)),
                trade_count: Some(150),
                taker_buy_base_volume: Some(dec!(750)),
                taker_buy_quote_volume: Some(dec!(75000)),
            },
            OhlcData {
                timestamp: Utc::now(),
                open: dec!(105),
                high: dec!(110),
                low: dec!(103),
                close: dec!(108),
                volume: dec!(1200),
                quote_volume: Some(dec!(120000)),
                trade_count: Some(120),
                taker_buy_base_volume: Some(dec!(600)),
                taker_buy_quote_volume: Some(dec!(60000)),
            },
        ];

        for ohlc in &test_data {
            let result = klinger.update(ohlc);
            assert!(result.is_ok());
        }

        // Should have some Klinger values calculated
        assert!(!klinger.klinger_values.is_empty());
        assert!(!klinger.volume_force_history.is_empty());
    }

    #[test]
    fn test_klinger_signals() {
        let config = KlingerConfig {
            fast_period: 3,
            slow_period: 5,
            signal_period: 2,
            ..Default::default()
        };
        let mut klinger = KlingerOscillator::new(config).unwrap();

        // Create test data with clear uptrend and high volume
        let test_data = vec![
            OhlcData {
                timestamp: Utc::now(),
                open: dec!(100),
                high: dec!(102),
                low: dec!(99),
                close: dec!(101),
                volume: dec!(1000),
                quote_volume: Some(dec!(100000)),
                trade_count: Some(100),
                taker_buy_base_volume: Some(dec!(500)),
                taker_buy_quote_volume: Some(dec!(50000)),
            },
            OhlcData {
                timestamp: Utc::now(),
                open: dec!(101),
                high: dec!(104),
                low: dec!(100),
                close: dec!(103),
                volume: dec!(1500),
                quote_volume: Some(dec!(150000)),
                trade_count: Some(150),
                taker_buy_base_volume: Some(dec!(750)),
                taker_buy_quote_volume: Some(dec!(75000)),
            },
            OhlcData {
                timestamp: Utc::now(),
                open: dec!(103),
                high: dec!(106),
                low: dec!(102),
                close: dec!(105),
                volume: dec!(2000),
                quote_volume: Some(dec!(200000)),
                trade_count: Some(200),
                taker_buy_base_volume: Some(dec!(1000)),
                taker_buy_quote_volume: Some(dec!(100000)),
            },
        ];

        for ohlc in &test_data {
            let _ = klinger.update(ohlc);
        }

        // Test signal generation
        if klinger.is_ready() {
            let signal = klinger.generate_signal(Utc::now());
            assert!(signal.is_ok());
        }
    }

    #[test]
    fn test_klinger_volume_force() {
        let mut klinger = KlingerOscillator::default();

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

        let volume_force = klinger.calculate_volume_force(&ohlc);
        assert!(volume_force.is_ok());

        // Volume force should be non-zero for non-zero volume
        let vf = volume_force.unwrap();
        assert_ne!(vf, dec!(0));
    }
}
