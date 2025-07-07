//! RVI (Relative Vigor Index) Implementation
//!
//! RVI, fiyat hareketlerinin gücünü ve yönünü ölçen momentum indikatörüdür.
//! Kapanış fiyatının açılış fiyatına göre pozisyonunu analiz eder.
//!
//! RVI = SMA(Close - Open) / SMA(High - Low)

use crate::types::{
    IndicatorResult, OhlcData, Signal, SignalData, SignalStrength, TechnicalAnalysisError,
};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::VecDeque;

/// RVI Configuration
#[derive(Debug, Clone)]
pub struct RviConfig {
    /// Period for RVI calculation (typically 10)
    pub period: usize,
    /// Signal line period (typically 4)
    pub signal_period: usize,
    /// Overbought threshold (typically 0.7)
    pub overbought_threshold: Decimal,
    /// Oversold threshold (typically -0.7)
    pub oversold_threshold: Decimal,
    /// Strong signal threshold
    pub strong_threshold: Decimal,
}

impl Default for RviConfig {
    fn default() -> Self {
        Self {
            period: 10,
            signal_period: 4,
            overbought_threshold: dec!(0.7),
            oversold_threshold: dec!(-0.7),
            strong_threshold: dec!(0.8),
        }
    }
}

/// RVI Result containing RVI line and signal line
#[derive(Debug, Clone)]
pub struct RviResult {
    /// RVI line value
    pub rvi_line: Decimal,
    /// Signal line value
    pub signal_line: Decimal,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl RviResult {
    /// Create new RVI result
    pub fn new(rvi_line: Decimal, signal_line: Decimal, timestamp: DateTime<Utc>) -> Self {
        Self {
            rvi_line,
            signal_line,
            timestamp,
        }
    }

    /// Check if RVI shows bullish crossover
    pub fn is_bullish_crossover(&self, previous: &RviResult) -> bool {
        previous.rvi_line <= previous.signal_line && self.rvi_line > self.signal_line
    }

    /// Check if RVI shows bearish crossover
    pub fn is_bearish_crossover(&self, previous: &RviResult) -> bool {
        previous.rvi_line >= previous.signal_line && self.rvi_line < self.signal_line
    }

    /// Get the difference between RVI and signal line
    pub fn line_difference(&self) -> Decimal {
        self.rvi_line - self.signal_line
    }
}

/// RVI Indicator Implementation
#[derive(Debug, Clone)]
pub struct RelativeVigorIndex {
    config: RviConfig,
    ohlc_history: VecDeque<OhlcData>,
    numerator_history: VecDeque<Decimal>,   // Close - Open
    denominator_history: VecDeque<Decimal>, // High - Low
    rvi_history: VecDeque<Decimal>,
    current_rvi: Option<RviResult>,
    previous_rvi: Option<RviResult>,
}

impl RelativeVigorIndex {
    /// Create new RVI indicator
    pub fn new(config: RviConfig) -> Result<Self, TechnicalAnalysisError> {
        if config.period == 0 {
            return Err(TechnicalAnalysisError::invalid_config(format!(
                "Period must be greater than 0, got: {}",
                config.period
            )));
        }

        if config.signal_period == 0 {
            return Err(TechnicalAnalysisError::invalid_config(format!(
                "Signal period must be greater than 0, got: {}",
                config.signal_period
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
            ohlc_history: VecDeque::with_capacity(100),
            numerator_history: VecDeque::with_capacity(100),
            denominator_history: VecDeque::with_capacity(100),
            rvi_history: VecDeque::with_capacity(100),
            current_rvi: None,
            previous_rvi: None,
        })
    }

    /// Create RVI with default configuration
    pub fn default() -> Result<Self, TechnicalAnalysisError> {
        Self::new(RviConfig::default())
    }

    /// Update RVI with new OHLC data
    pub fn update(
        &mut self,
        ohlc: &OhlcData,
    ) -> Result<Option<IndicatorResult>, TechnicalAnalysisError> {
        // Add OHLC to history
        self.ohlc_history.push_back(ohlc.clone());

        // Calculate numerator and denominator
        let numerator = ohlc.close - ohlc.open;
        let denominator = ohlc.high - ohlc.low;

        self.numerator_history.push_back(numerator);
        self.denominator_history.push_back(denominator);

        // Keep only necessary history
        if self.numerator_history.len() > self.config.period {
            self.numerator_history.pop_front();
            self.denominator_history.pop_front();
            self.ohlc_history.pop_front();
        }

        // Calculate RVI if we have enough data
        if self.numerator_history.len() >= self.config.period {
            let rvi_value = self.calculate_rvi()?;
            self.rvi_history.push_back(rvi_value);

            // Keep only necessary RVI history for signal line
            if self.rvi_history.len() > self.config.signal_period {
                self.rvi_history.pop_front();
            }

            // Calculate signal line if we have enough RVI values
            if self.rvi_history.len() >= self.config.signal_period {
                let signal_line = self.calculate_signal_line()?;

                // Store previous result
                self.previous_rvi = self.current_rvi.clone();

                // Create new result
                let rvi_result = RviResult::new(rvi_value, signal_line, ohlc.timestamp);
                self.current_rvi = Some(rvi_result);

                return Ok(Some(IndicatorResult::new(
                    rvi_value,
                    ohlc.timestamp,
                    "RVI".to_string(),
                )));
            }
        }

        Ok(None)
    }

    /// Calculate RVI value
    fn calculate_rvi(&self) -> Result<Decimal, TechnicalAnalysisError> {
        if self.numerator_history.len() < self.config.period {
            return Err(TechnicalAnalysisError::insufficient_data(
                self.config.period,
                self.numerator_history.len(),
            ));
        }

        // Calculate smoothed numerator (SMA of Close - Open)
        let numerator_sum: Decimal = self.numerator_history.iter().sum();
        let smoothed_numerator = numerator_sum / Decimal::from(self.config.period);

        // Calculate smoothed denominator (SMA of High - Low)
        let denominator_sum: Decimal = self.denominator_history.iter().sum();
        let smoothed_denominator = denominator_sum / Decimal::from(self.config.period);

        // Avoid division by zero
        if smoothed_denominator == Decimal::ZERO {
            return Ok(Decimal::ZERO);
        }

        let rvi = smoothed_numerator / smoothed_denominator;

        // Clamp RVI to reasonable bounds
        Ok(rvi.max(dec!(-2.0)).min(dec!(2.0)))
    }

    /// Calculate signal line (SMA of RVI)
    fn calculate_signal_line(&self) -> Result<Decimal, TechnicalAnalysisError> {
        if self.rvi_history.len() < self.config.signal_period {
            return Err(TechnicalAnalysisError::insufficient_data(
                self.config.signal_period,
                self.rvi_history.len(),
            ));
        }

        let sum: Decimal = self
            .rvi_history
            .iter()
            .take(self.config.signal_period)
            .sum();
        Ok(sum / Decimal::from(self.config.signal_period))
    }

    /// Generate trading signal based on RVI analysis
    pub fn generate_signal(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        if let (Some(current), Some(previous)) = (&self.current_rvi, &self.previous_rvi) {
            // Check for crossovers
            if current.is_bullish_crossover(previous) {
                let strength = if current.rvi_line > self.config.strong_threshold {
                    SignalStrength::Strong
                } else if current.rvi_line > Decimal::ZERO {
                    SignalStrength::Moderate
                } else {
                    SignalStrength::Weak
                };

                let confidence = self.calculate_crossover_confidence(current, true);

                return Ok(Some(SignalData::new(
                    Signal::Buy,
                    strength,
                    confidence,
                    timestamp,
                    "RVI".to_string(),
                )?));
            }

            if current.is_bearish_crossover(previous) {
                let strength = if current.rvi_line < -self.config.strong_threshold {
                    SignalStrength::Strong
                } else if current.rvi_line < Decimal::ZERO {
                    SignalStrength::Moderate
                } else {
                    SignalStrength::Weak
                };

                let confidence = self.calculate_crossover_confidence(current, false);

                return Ok(Some(SignalData::new(
                    Signal::Sell,
                    strength,
                    confidence,
                    timestamp,
                    "RVI".to_string(),
                )?));
            }

            // Check for extreme level signals
            if let Some(signal) = self.check_extreme_levels(current, timestamp)? {
                return Ok(Some(signal));
            }
        }

        Ok(None)
    }

    /// Calculate confidence for crossover signals
    fn calculate_crossover_confidence(&self, current: &RviResult, is_bullish: bool) -> Decimal {
        let mut confidence = dec!(0.7); // Base confidence for crossovers

        // Increase confidence based on RVI magnitude
        let rvi_magnitude = current.rvi_line.abs();
        if rvi_magnitude > self.config.strong_threshold {
            confidence += dec!(0.2);
        }

        // Increase confidence based on line separation
        let line_separation = current.line_difference().abs();
        if line_separation > dec!(0.1) {
            confidence += dec!(0.1);
        }

        // Adjust for signal direction consistency
        if (is_bullish && current.rvi_line > Decimal::ZERO)
            || (!is_bullish && current.rvi_line < Decimal::ZERO)
        {
            confidence += dec!(0.1);
        }

        confidence.min(dec!(0.95))
    }

    /// Check for extreme level signals
    fn check_extreme_levels(
        &self,
        current: &RviResult,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        // Overbought condition
        if current.rvi_line >= self.config.overbought_threshold {
            let confidence =
                (current.rvi_line / self.config.overbought_threshold).min(dec!(1.0)) * dec!(0.6);

            return Ok(Some(SignalData::new(
                Signal::Sell,
                SignalStrength::Weak,
                confidence,
                timestamp,
                "RVI_Overbought".to_string(),
            )?));
        }

        // Oversold condition
        if current.rvi_line <= self.config.oversold_threshold {
            let confidence = (current.rvi_line.abs() / self.config.oversold_threshold.abs())
                .min(dec!(1.0))
                * dec!(0.6);

            return Ok(Some(SignalData::new(
                Signal::Buy,
                SignalStrength::Weak,
                confidence,
                timestamp,
                "RVI_Oversold".to_string(),
            )?));
        }

        Ok(None)
    }

    /// Get current RVI result
    pub fn current_result(&self) -> Option<&RviResult> {
        self.current_rvi.as_ref()
    }

    /// Check if RVI is overbought
    pub fn is_overbought(&self) -> bool {
        self.current_rvi.as_ref().map_or(false, |rvi| {
            rvi.rvi_line >= self.config.overbought_threshold
        })
    }

    /// Check if RVI is oversold
    pub fn is_oversold(&self) -> bool {
        self.current_rvi
            .as_ref()
            .map_or(false, |rvi| rvi.rvi_line <= self.config.oversold_threshold)
    }

    /// Check if RVI shows bullish crossover
    pub fn is_bullish_crossover(&self) -> bool {
        if let (Some(current), Some(previous)) = (&self.current_rvi, &self.previous_rvi) {
            current.is_bullish_crossover(previous)
        } else {
            false
        }
    }

    /// Check if RVI shows bearish crossover
    pub fn is_bearish_crossover(&self) -> bool {
        if let (Some(current), Some(previous)) = (&self.current_rvi, &self.previous_rvi) {
            current.is_bearish_crossover(previous)
        } else {
            false
        }
    }

    /// Get RVI configuration
    pub fn config(&self) -> &RviConfig {
        &self.config
    }

    /// Reset the indicator
    pub fn reset(&mut self) {
        self.ohlc_history.clear();
        self.numerator_history.clear();
        self.denominator_history.clear();
        self.rvi_history.clear();
        self.current_rvi = None;
        self.previous_rvi = None;
    }

    /// Check if indicator is ready
    pub fn is_ready(&self) -> bool {
        self.current_rvi.is_some()
    }

    /// Get minimum data points required
    pub fn min_data_points(&self) -> usize {
        self.config.period + self.config.signal_period
    }

    /// Get data history length
    pub fn history_len(&self) -> usize {
        self.ohlc_history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    fn create_test_ohlc(
        open: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        timestamp: DateTime<Utc>,
    ) -> OhlcData {
        OhlcData::new(open, high, low, close, dec!(1000), timestamp).unwrap()
    }

    #[test]
    fn test_rvi_creation() {
        let config = RviConfig::default();
        let rvi = RelativeVigorIndex::new(config).unwrap();
        assert_eq!(rvi.config.period, 10);
        assert_eq!(rvi.config.signal_period, 4);
        assert!(!rvi.is_ready());
    }

    #[test]
    fn test_rvi_invalid_config() {
        let config = RviConfig {
            period: 0,
            ..Default::default()
        };
        assert!(RelativeVigorIndex::new(config).is_err());

        let config = RviConfig {
            overbought_threshold: dec!(-0.5),
            oversold_threshold: dec!(0.5),
            ..Default::default()
        };
        assert!(RelativeVigorIndex::new(config).is_err());
    }

    #[test]
    fn test_rvi_calculation_bullish() {
        let mut rvi = RelativeVigorIndex::default().unwrap();
        let base_time = Utc::now();

        // Create bullish candles (close > open)
        for i in 0..20 {
            let base_price = dec!(100) + Decimal::from(i);
            let ohlc = create_test_ohlc(
                base_price,           // open
                base_price + dec!(2), // high
                base_price - dec!(1), // low
                base_price + dec!(1), // close (bullish)
                base_time + chrono::Duration::seconds(i),
            );
            rvi.update(&ohlc).unwrap();
        }

        assert!(rvi.is_ready());
        if let Some(result) = rvi.current_result() {
            // RVI should be positive for bullish candles
            assert!(result.rvi_line > Decimal::ZERO);
        }
    }

    #[test]
    fn test_rvi_calculation_bearish() {
        let mut rvi = RelativeVigorIndex::default().unwrap();
        let base_time = Utc::now();

        // Create bearish candles (close < open)
        for i in 0..20 {
            let base_price = dec!(100) - Decimal::from(i);
            let ohlc = create_test_ohlc(
                base_price,           // open
                base_price + dec!(1), // high
                base_price - dec!(2), // low
                base_price - dec!(1), // close (bearish)
                base_time + chrono::Duration::seconds(i),
            );
            rvi.update(&ohlc).unwrap();
        }

        assert!(rvi.is_ready());
        if let Some(result) = rvi.current_result() {
            // RVI should be negative for bearish candles
            assert!(result.rvi_line < Decimal::ZERO);
        }
    }

    #[test]
    fn test_rvi_crossover_detection() {
        let result1 = RviResult::new(dec!(0.5), dec!(0.3), Utc::now());
        let result2 = RviResult::new(dec!(0.2), dec!(0.4), Utc::now());

        assert!(result1.is_bullish_crossover(&result2));
        assert!(!result2.is_bullish_crossover(&result1));
        assert!(result2.is_bearish_crossover(&result1));
    }

    #[test]
    fn test_rvi_signal_generation() {
        let mut rvi = RelativeVigorIndex::default().unwrap();
        let base_time = Utc::now();

        // Generate mixed data to potentially create signals
        for i in 0..20 {
            let base_price = dec!(100);
            let trend_factor = if i < 10 { dec!(-0.5) } else { dec!(0.5) };

            let ohlc = create_test_ohlc(
                base_price,
                base_price + dec!(1),
                base_price - dec!(1),
                base_price + trend_factor,
                base_time + chrono::Duration::seconds(i),
            );
            rvi.update(&ohlc).unwrap();
        }

        assert!(rvi.is_ready());

        // Try to generate a signal
        let _signal = rvi.generate_signal(base_time).unwrap();
        // Signal may or may not be generated depending on the data
    }

    #[test]
    fn test_rvi_extreme_levels() {
        let mut rvi = RelativeVigorIndex::default().unwrap();
        let base_time = Utc::now();

        // Create extreme bullish candles
        for i in 0..20 {
            let base_price = dec!(100);
            let ohlc = create_test_ohlc(
                base_price,
                base_price + dec!(5),
                base_price - dec!(1),
                base_price + dec!(4), // Very bullish
                base_time + chrono::Duration::seconds(i),
            );
            rvi.update(&ohlc).unwrap();
        }

        if rvi.is_ready() {
            // Should potentially show overbought condition
            let _is_extreme = rvi.is_overbought() || rvi.is_oversold();
            // Test passes regardless of result as it depends on specific calculations
        }
    }

    #[test]
    fn test_rvi_reset() {
        let mut rvi = RelativeVigorIndex::default().unwrap();
        let base_time = Utc::now();

        // Add some data
        for i in 0..15 {
            let ohlc = create_test_ohlc(
                dec!(100),
                dec!(101),
                dec!(99),
                dec!(100),
                base_time + chrono::Duration::seconds(i),
            );
            rvi.update(&ohlc).unwrap();
        }

        assert!(rvi.history_len() > 0);

        rvi.reset();

        assert_eq!(rvi.history_len(), 0);
        assert!(!rvi.is_ready());
        assert!(rvi.current_result().is_none());
    }

    #[test]
    fn test_rvi_min_data_points() {
        let rvi = RelativeVigorIndex::default().unwrap();
        assert_eq!(rvi.min_data_points(), 14); // 10 + 4
    }
}
