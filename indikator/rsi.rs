//! RSI (Relative Strength Index) Implementation
//!
//! RSI, momentum oscillatörüdür ve 0-100 arasında değer alır.
//! 70 üzeri overbought, 30 altı oversold olarak kabul edilir.
//!
//! Formula:
//! RSI = 100 - (100 / (1 + RS))
//! RS = Average Gain / Average Loss

use crate::types::{
    IndicatorResult, OhlcData, Signal, SignalData, SignalStrength, TechnicalAnalysisError,
};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::VecDeque;

/// RSI Configuration
#[derive(Debug, Clone)]
pub struct RsiConfig {
    /// Period for RSI calculation (typically 14)
    pub period: usize,
    /// Overbought threshold (typically 70)
    pub overbought_threshold: Decimal,
    /// Oversold threshold (typically 30)
    pub oversold_threshold: Decimal,
    /// Strong signal threshold (typically 80/20)
    pub strong_overbought_threshold: Decimal,
    /// Strong oversold threshold (typically 20)
    pub strong_oversold_threshold: Decimal,
}

impl Default for RsiConfig {
    fn default() -> Self {
        Self {
            period: 14,
            overbought_threshold: dec!(70),
            oversold_threshold: dec!(30),
            strong_overbought_threshold: dec!(80),
            strong_oversold_threshold: dec!(20),
        }
    }
}

/// RSI Indicator Implementation
#[derive(Debug, Clone)]
pub struct RelativeStrengthIndex {
    config: RsiConfig,
    price_history: VecDeque<Decimal>,
    gains: VecDeque<Decimal>,
    losses: VecDeque<Decimal>,
    avg_gain: Option<Decimal>,
    avg_loss: Option<Decimal>,
    current_rsi: Option<Decimal>,
    previous_close: Option<Decimal>,
}

impl RelativeStrengthIndex {
    /// Create new RSI indicator
    pub fn new(config: RsiConfig) -> Result<Self, TechnicalAnalysisError> {
        if config.period == 0 {
            return Err(TechnicalAnalysisError::invalid_config(format!(
                "Period must be greater than 0, got: {}",
                config.period
            )));
        }

        if config.overbought_threshold <= config.oversold_threshold {
            return Err(TechnicalAnalysisError::invalid_config(format!(
                "Overbought threshold ({}) must be greater than oversold threshold ({})",
                config.overbought_threshold, config.oversold_threshold
            )));
        }

        Ok(Self {
            config,
            price_history: VecDeque::with_capacity(100),
            gains: VecDeque::with_capacity(100),
            losses: VecDeque::with_capacity(100),
            avg_gain: None,
            avg_loss: None,
            current_rsi: None,
            previous_close: None,
        })
    }

    /// Create RSI with default configuration
    pub fn default() -> Result<Self, TechnicalAnalysisError> {
        Self::new(RsiConfig::default())
    }

    /// Update RSI with new OHLC data
    pub fn update(
        &mut self,
        ohlc: &OhlcData,
    ) -> Result<Option<IndicatorResult>, TechnicalAnalysisError> {
        // Add current close price to history
        self.price_history.push_back(ohlc.close);

        // Keep only necessary history
        if self.price_history.len() > self.config.period + 1 {
            self.price_history.pop_front();
        }

        // Calculate price change if we have previous close
        if let Some(prev_close) = self.previous_close {
            let change = ohlc.close - prev_close;

            if change > Decimal::ZERO {
                self.gains.push_back(change);
                self.losses.push_back(Decimal::ZERO);
            } else {
                self.gains.push_back(Decimal::ZERO);
                self.losses.push_back(-change);
            }

            // Keep only necessary history
            if self.gains.len() > self.config.period {
                self.gains.pop_front();
                self.losses.pop_front();
            }
        }

        self.previous_close = Some(ohlc.close);

        // Calculate RSI if we have enough data
        if self.gains.len() >= self.config.period {
            self.current_rsi = Some(self.calculate_rsi()?);

            Ok(Some(IndicatorResult::new(
                self.current_rsi.unwrap(),
                ohlc.timestamp,
                "RSI".to_string(),
            )))
        } else {
            Ok(None)
        }
    }

    /// Calculate RSI value
    fn calculate_rsi(&mut self) -> Result<Decimal, TechnicalAnalysisError> {
        if self.gains.len() < self.config.period {
            return Err(TechnicalAnalysisError::insufficient_data(
                self.config.period,
                self.gains.len(),
            ));
        }

        // Calculate average gain and loss
        if self.avg_gain.is_none() || self.avg_loss.is_none() {
            // First calculation - simple average
            let sum_gain: Decimal = self.gains.iter().sum();
            let sum_loss: Decimal = self.losses.iter().sum();

            self.avg_gain = Some(sum_gain / Decimal::from(self.config.period));
            self.avg_loss = Some(sum_loss / Decimal::from(self.config.period));
        } else {
            // Subsequent calculations - smoothed average (Wilder's smoothing)
            let latest_gain = *self.gains.back().unwrap();
            let latest_loss = *self.losses.back().unwrap();

            let period_decimal = Decimal::from(self.config.period);

            self.avg_gain = Some(
                (self.avg_gain.unwrap() * (period_decimal - Decimal::ONE) + latest_gain)
                    / period_decimal,
            );
            self.avg_loss = Some(
                (self.avg_loss.unwrap() * (period_decimal - Decimal::ONE) + latest_loss)
                    / period_decimal,
            );
        }

        let avg_gain = self.avg_gain.unwrap();
        let avg_loss = self.avg_loss.unwrap();

        // Calculate RSI
        if avg_loss == Decimal::ZERO {
            Ok(dec!(100)) // Maximum RSI when no losses
        } else {
            let rs = avg_gain / avg_loss;
            let rsi = dec!(100) - (dec!(100) / (Decimal::ONE + rs));
            Ok(rsi)
        }
    }

    /// Generate trading signal based on RSI value
    pub fn generate_signal(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        if let Some(rsi_value) = self.current_rsi {
            let (signal, strength) = if rsi_value >= self.config.strong_overbought_threshold {
                (Signal::Sell, SignalStrength::Strong)
            } else if rsi_value >= self.config.overbought_threshold {
                (Signal::Sell, SignalStrength::Weak)
            } else if rsi_value <= self.config.strong_oversold_threshold {
                (Signal::Buy, SignalStrength::Strong)
            } else if rsi_value <= self.config.oversold_threshold {
                (Signal::Buy, SignalStrength::Weak)
            } else {
                return Ok(None); // No signal in neutral zone
            };

            // Calculate confidence based on distance from threshold
            let confidence = self.calculate_confidence(rsi_value);

            Ok(Some(SignalData::new(
                signal,
                strength,
                confidence,
                timestamp,
                "RSI".to_string(),
            )?))
        } else {
            Ok(None)
        }
    }

    /// Calculate signal confidence based on RSI value
    fn calculate_confidence(&self, rsi_value: Decimal) -> Decimal {
        if rsi_value >= self.config.overbought_threshold {
            // Overbought zone - higher RSI = higher confidence for sell signal
            let max_confidence_range = dec!(100) - self.config.overbought_threshold;
            let current_range = rsi_value - self.config.overbought_threshold;
            (current_range / max_confidence_range).min(Decimal::ONE)
        } else if rsi_value <= self.config.oversold_threshold {
            // Oversold zone - lower RSI = higher confidence for buy signal
            let max_confidence_range = self.config.oversold_threshold;
            let current_range = self.config.oversold_threshold - rsi_value;
            (current_range / max_confidence_range).min(Decimal::ONE)
        } else {
            // Neutral zone
            dec!(0.5)
        }
    }

    /// Get current RSI value
    pub fn current_value(&self) -> Option<Decimal> {
        self.current_rsi
    }

    /// Check if RSI is in overbought condition
    pub fn is_overbought(&self) -> bool {
        self.current_rsi
            .map_or(false, |rsi| rsi >= self.config.overbought_threshold)
    }

    /// Check if RSI is in oversold condition
    pub fn is_oversold(&self) -> bool {
        self.current_rsi
            .map_or(false, |rsi| rsi <= self.config.oversold_threshold)
    }

    /// Check if RSI is in strong overbought condition
    pub fn is_strong_overbought(&self) -> bool {
        self.current_rsi
            .map_or(false, |rsi| rsi >= self.config.strong_overbought_threshold)
    }

    /// Check if RSI is in strong oversold condition
    pub fn is_strong_oversold(&self) -> bool {
        self.current_rsi
            .map_or(false, |rsi| rsi <= self.config.strong_oversold_threshold)
    }

    /// Get RSI configuration
    pub fn config(&self) -> &RsiConfig {
        &self.config
    }

    /// Reset the indicator
    pub fn reset(&mut self) {
        self.price_history.clear();
        self.gains.clear();
        self.losses.clear();
        self.avg_gain = None;
        self.avg_loss = None;
        self.current_rsi = None;
        self.previous_close = None;
    }

    /// Get data history length
    pub fn history_len(&self) -> usize {
        self.price_history.len()
    }

    /// Check if indicator is ready (has enough data)
    pub fn is_ready(&self) -> bool {
        self.gains.len() >= self.config.period
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    fn create_test_ohlc(close: Decimal, timestamp: DateTime<Utc>) -> OhlcData {
        OhlcData::new(close, close, close, close, dec!(1000), timestamp).unwrap()
    }

    #[test]
    fn test_rsi_creation() {
        let config = RsiConfig::default();
        let rsi = RelativeStrengthIndex::new(config).unwrap();
        assert_eq!(rsi.config.period, 14);
        assert!(!rsi.is_ready());
    }

    #[test]
    fn test_rsi_invalid_period() {
        let config = RsiConfig {
            period: 0,
            ..Default::default()
        };
        assert!(RelativeStrengthIndex::new(config).is_err());
    }

    #[test]
    fn test_rsi_invalid_thresholds() {
        let config = RsiConfig {
            overbought_threshold: dec!(30),
            oversold_threshold: dec!(70),
            ..Default::default()
        };
        assert!(RelativeStrengthIndex::new(config).is_err());
    }

    #[test]
    fn test_rsi_calculation_uptrend() {
        let mut rsi = RelativeStrengthIndex::default().unwrap();
        let base_time = Utc::now();

        // Simulate uptrend - prices going up
        let prices = [
            dec!(100),
            dec!(101),
            dec!(102),
            dec!(103),
            dec!(104),
            dec!(105),
            dec!(106),
            dec!(107),
            dec!(108),
            dec!(109),
            dec!(110),
            dec!(111),
            dec!(112),
            dec!(113),
            dec!(114),
            dec!(115),
            dec!(116),
            dec!(117),
            dec!(118),
            dec!(119),
        ];

        let mut _last_result = None;
        for (i, &price) in prices.iter().enumerate() {
            let ohlc = create_test_ohlc(price, base_time + chrono::Duration::seconds(i as i64));
            _last_result = rsi.update(&ohlc).unwrap();
        }

        assert!(rsi.is_ready());
        let rsi_value = rsi.current_value().unwrap();
        assert!(rsi_value > dec!(50)); // Should be high in uptrend
        assert!(rsi_value <= dec!(100));
    }

    #[test]
    fn test_rsi_calculation_downtrend() {
        let mut rsi = RelativeStrengthIndex::default().unwrap();
        let base_time = Utc::now();

        // Simulate downtrend - prices going down
        let prices = [
            dec!(100),
            dec!(99),
            dec!(98),
            dec!(97),
            dec!(96),
            dec!(95),
            dec!(94),
            dec!(93),
            dec!(92),
            dec!(91),
            dec!(90),
            dec!(89),
            dec!(88),
            dec!(87),
            dec!(86),
            dec!(85),
            dec!(84),
            dec!(83),
            dec!(82),
            dec!(81),
        ];

        let mut _last_result = None;
        for (i, &price) in prices.iter().enumerate() {
            let ohlc = create_test_ohlc(price, base_time + chrono::Duration::seconds(i as i64));
            _last_result = rsi.update(&ohlc).unwrap();
        }

        assert!(rsi.is_ready());
        let rsi_value = rsi.current_value().unwrap();
        assert!(rsi_value < dec!(50)); // Should be low in downtrend
        assert!(rsi_value >= dec!(0));
    }

    #[test]
    fn test_rsi_overbought_signal() {
        let mut rsi = RelativeStrengthIndex::default().unwrap();
        let base_time = Utc::now();

        // Create strong uptrend to reach overbought
        let prices = [
            dec!(100),
            dec!(105),
            dec!(110),
            dec!(115),
            dec!(120),
            dec!(125),
            dec!(130),
            dec!(135),
            dec!(140),
            dec!(145),
            dec!(150),
            dec!(155),
            dec!(160),
            dec!(165),
            dec!(170),
            dec!(175),
            dec!(180),
            dec!(185),
            dec!(190),
            dec!(195),
        ];

        for (i, &price) in prices.iter().enumerate() {
            let ohlc = create_test_ohlc(price, base_time + chrono::Duration::seconds(i as i64));
            rsi.update(&ohlc).unwrap();
        }

        assert!(rsi.is_overbought());

        let signal = rsi.generate_signal(base_time).unwrap();
        assert!(signal.is_some());
        let signal = signal.unwrap();
        assert_eq!(signal.signal, Signal::Sell);
    }

    #[test]
    fn test_rsi_oversold_signal() {
        let mut rsi = RelativeStrengthIndex::default().unwrap();
        let base_time = Utc::now();

        // Create strong downtrend to reach oversold
        let prices = [
            dec!(100),
            dec!(95),
            dec!(90),
            dec!(85),
            dec!(80),
            dec!(75),
            dec!(70),
            dec!(65),
            dec!(60),
            dec!(55),
            dec!(50),
            dec!(45),
            dec!(40),
            dec!(35),
            dec!(30),
            dec!(25),
            dec!(20),
            dec!(15),
            dec!(10),
            dec!(5),
        ];

        for (i, &price) in prices.iter().enumerate() {
            let ohlc = create_test_ohlc(price, base_time + chrono::Duration::seconds(i as i64));
            rsi.update(&ohlc).unwrap();
        }

        assert!(rsi.is_oversold());

        let signal = rsi.generate_signal(base_time).unwrap();
        assert!(signal.is_some());
        let signal = signal.unwrap();
        assert_eq!(signal.signal, Signal::Buy);
    }

    #[test]
    fn test_rsi_neutral_zone() {
        let mut rsi = RelativeStrengthIndex::default().unwrap();
        let base_time = Utc::now();

        // Create sideways movement to stay in neutral zone
        let prices = [
            dec!(100),
            dec!(101),
            dec!(99),
            dec!(100),
            dec!(101),
            dec!(99),
            dec!(100),
            dec!(101),
            dec!(99),
            dec!(100),
            dec!(101),
            dec!(99),
            dec!(100),
            dec!(101),
            dec!(99),
            dec!(100),
            dec!(101),
            dec!(99),
            dec!(100),
            dec!(101),
        ];

        for (i, &price) in prices.iter().enumerate() {
            let ohlc = create_test_ohlc(price, base_time + chrono::Duration::seconds(i as i64));
            rsi.update(&ohlc).unwrap();
        }

        assert!(!rsi.is_overbought());
        assert!(!rsi.is_oversold());

        let signal = rsi.generate_signal(base_time).unwrap();
        assert!(signal.is_none()); // No signal in neutral zone
    }

    #[test]
    fn test_rsi_reset() {
        let mut rsi = RelativeStrengthIndex::default().unwrap();
        let base_time = Utc::now();

        // Add some data
        for i in 0..10 {
            let ohlc = create_test_ohlc(
                dec!(100) + Decimal::from(i),
                base_time + chrono::Duration::seconds(i),
            );
            rsi.update(&ohlc).unwrap();
        }

        assert!(rsi.history_len() > 0);

        rsi.reset();

        assert_eq!(rsi.history_len(), 0);
        assert!(!rsi.is_ready());
        assert!(rsi.current_value().is_none());
    }
}
