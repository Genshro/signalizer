//! CCI (Commodity Channel Index) Implementation
//!
//! CCI, fiyatın istatistiksel ortalamadan ne kadar saptığını ölçen momentum indikatörüdür.
//!
//! Typical Price = (High + Low + Close) / 3
//! CCI = (Typical Price - SMA of Typical Price) / (0.015 × Mean Deviation)

use crate::types::{
    IndicatorResult, OhlcData, Signal, SignalData, SignalStrength, TechnicalAnalysisError,
};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::VecDeque;

/// CCI Configuration
#[derive(Debug, Clone)]
pub struct CciConfig {
    /// Period for CCI calculation (typically 20)
    pub period: usize,
    /// Overbought threshold (typically +100)
    pub overbought_threshold: Decimal,
    /// Oversold threshold (typically -100)
    pub oversold_threshold: Decimal,
    /// Extreme overbought threshold (typically +200)
    pub extreme_overbought_threshold: Decimal,
    /// Extreme oversold threshold (typically -200)
    pub extreme_oversold_threshold: Decimal,
    /// CCI constant factor (typically 0.015)
    pub constant_factor: Decimal,
}

impl Default for CciConfig {
    fn default() -> Self {
        Self {
            period: 20,
            overbought_threshold: dec!(100),
            oversold_threshold: dec!(-100),
            extreme_overbought_threshold: dec!(200),
            extreme_oversold_threshold: dec!(-200),
            constant_factor: dec!(0.015),
        }
    }
}

/// CCI Indicator Implementation
#[derive(Debug, Clone)]
pub struct CommodityChannelIndex {
    config: CciConfig,
    typical_prices: VecDeque<Decimal>,
    current_cci: Option<Decimal>,
    previous_cci: Option<Decimal>,
}

impl CommodityChannelIndex {
    /// Create new CCI indicator
    pub fn new(config: CciConfig) -> Result<Self, TechnicalAnalysisError> {
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

        if config.constant_factor <= Decimal::ZERO {
            return Err(TechnicalAnalysisError::invalid_config(format!(
                "Constant factor must be positive, got: {}",
                config.constant_factor
            )));
        }

        Ok(Self {
            config,
            typical_prices: VecDeque::with_capacity(100),
            current_cci: None,
            previous_cci: None,
        })
    }

    /// Create CCI with default configuration
    pub fn default() -> Result<Self, TechnicalAnalysisError> {
        Self::new(CciConfig::default())
    }

    /// Update CCI with new OHLC data
    pub fn update(
        &mut self,
        ohlc: &OhlcData,
    ) -> Result<Option<IndicatorResult>, TechnicalAnalysisError> {
        // Calculate typical price
        let typical_price = (ohlc.high + ohlc.low + ohlc.close) / dec!(3);
        self.typical_prices.push_back(typical_price);

        // Keep only necessary history
        if self.typical_prices.len() > self.config.period {
            self.typical_prices.pop_front();
        }

        // Calculate CCI if we have enough data
        if self.typical_prices.len() >= self.config.period {
            let cci_value = self.calculate_cci()?;

            // Store previous value
            self.previous_cci = self.current_cci;
            self.current_cci = Some(cci_value);

            return Ok(Some(IndicatorResult::new(
                cci_value,
                ohlc.timestamp,
                "CCI".to_string(),
            )));
        }

        Ok(None)
    }

    /// Calculate CCI value
    fn calculate_cci(&self) -> Result<Decimal, TechnicalAnalysisError> {
        if self.typical_prices.len() < self.config.period {
            return Err(TechnicalAnalysisError::insufficient_data(
                self.config.period,
                self.typical_prices.len(),
            ));
        }

        // Calculate SMA of typical prices
        let sum: Decimal = self.typical_prices.iter().sum();
        let sma = sum / Decimal::from(self.config.period);

        // Calculate mean deviation
        let mean_deviation = self.calculate_mean_deviation(sma)?;

        // Avoid division by zero
        if mean_deviation == Decimal::ZERO {
            return Ok(Decimal::ZERO);
        }

        // Get current typical price
        let current_typical_price = self
            .typical_prices
            .back()
            .ok_or_else(|| TechnicalAnalysisError::insufficient_data(1, 0))?;

        // Calculate CCI
        let cci = (current_typical_price - sma) / (self.config.constant_factor * mean_deviation);

        // Clamp CCI to reasonable bounds to prevent extreme values
        Ok(cci.max(dec!(-500)).min(dec!(500)))
    }

    /// Calculate mean deviation
    fn calculate_mean_deviation(&self, sma: Decimal) -> Result<Decimal, TechnicalAnalysisError> {
        let sum_deviations: Decimal = self
            .typical_prices
            .iter()
            .map(|&price| (price - sma).abs())
            .sum();

        Ok(sum_deviations / Decimal::from(self.config.period))
    }

    /// Generate trading signal based on CCI analysis
    pub fn generate_signal(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        if let (Some(current), Some(previous)) = (self.current_cci, self.previous_cci) {
            // Check for extreme level reversals
            if let Some(signal) = self.check_extreme_reversals(current, previous, timestamp)? {
                return Ok(Some(signal));
            }

            // Check for overbought/oversold conditions
            if let Some(signal) = self.check_standard_levels(current, previous, timestamp)? {
                return Ok(Some(signal));
            }

            // Check for zero line crossovers
            if let Some(signal) = self.check_zero_line_crossover(current, previous, timestamp)? {
                return Ok(Some(signal));
            }
        }

        Ok(None)
    }

    /// Check for extreme level reversals
    fn check_extreme_reversals(
        &self,
        current: Decimal,
        previous: Decimal,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        // Extreme overbought reversal
        if previous >= self.config.extreme_overbought_threshold && current < previous {
            let confidence =
                (previous / self.config.extreme_overbought_threshold).min(dec!(1.0)) * dec!(0.9);

            return Ok(Some(SignalData::new(
                Signal::Sell,
                SignalStrength::VeryStrong,
                confidence,
                timestamp,
                "CCI_Extreme_Overbought".to_string(),
            )?));
        }

        // Extreme oversold reversal
        if previous <= self.config.extreme_oversold_threshold && current > previous {
            let confidence = (previous.abs() / self.config.extreme_oversold_threshold.abs())
                .min(dec!(1.0))
                * dec!(0.9);

            return Ok(Some(SignalData::new(
                Signal::Buy,
                SignalStrength::VeryStrong,
                confidence,
                timestamp,
                "CCI_Extreme_Oversold".to_string(),
            )?));
        }

        Ok(None)
    }

    /// Check for standard overbought/oversold levels
    fn check_standard_levels(
        &self,
        current: Decimal,
        previous: Decimal,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        // Standard overbought condition
        if previous < self.config.overbought_threshold
            && current >= self.config.overbought_threshold
        {
            let confidence =
                (current / self.config.overbought_threshold).min(dec!(1.0)) * dec!(0.7);

            return Ok(Some(SignalData::new(
                Signal::Sell,
                SignalStrength::Moderate,
                confidence,
                timestamp,
                "CCI_Overbought".to_string(),
            )?));
        }

        // Standard oversold condition
        if previous > self.config.oversold_threshold && current <= self.config.oversold_threshold {
            let confidence =
                (current.abs() / self.config.oversold_threshold.abs()).min(dec!(1.0)) * dec!(0.7);

            return Ok(Some(SignalData::new(
                Signal::Buy,
                SignalStrength::Moderate,
                confidence,
                timestamp,
                "CCI_Oversold".to_string(),
            )?));
        }

        Ok(None)
    }

    /// Check for zero line crossover
    fn check_zero_line_crossover(
        &self,
        current: Decimal,
        previous: Decimal,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        // Bullish zero line crossover
        if previous <= Decimal::ZERO && current > Decimal::ZERO {
            let confidence = dec!(0.6); // Base confidence for zero line crossovers

            return Ok(Some(SignalData::new(
                Signal::Buy,
                SignalStrength::Weak,
                confidence,
                timestamp,
                "CCI_Zero_Cross_Up".to_string(),
            )?));
        }

        // Bearish zero line crossover
        if previous >= Decimal::ZERO && current < Decimal::ZERO {
            let confidence = dec!(0.6); // Base confidence for zero line crossovers

            return Ok(Some(SignalData::new(
                Signal::Sell,
                SignalStrength::Weak,
                confidence,
                timestamp,
                "CCI_Zero_Cross_Down".to_string(),
            )?));
        }

        Ok(None)
    }

    /// Get current CCI value
    pub fn current_value(&self) -> Option<Decimal> {
        self.current_cci
    }

    /// Get previous CCI value
    pub fn previous_value(&self) -> Option<Decimal> {
        self.previous_cci
    }

    /// Check if CCI is overbought
    pub fn is_overbought(&self) -> bool {
        self.current_cci
            .map_or(false, |cci| cci >= self.config.overbought_threshold)
    }

    /// Check if CCI is oversold
    pub fn is_oversold(&self) -> bool {
        self.current_cci
            .map_or(false, |cci| cci <= self.config.oversold_threshold)
    }

    /// Check if CCI is extremely overbought
    pub fn is_extremely_overbought(&self) -> bool {
        self.current_cci
            .map_or(false, |cci| cci >= self.config.extreme_overbought_threshold)
    }

    /// Check if CCI is extremely oversold
    pub fn is_extremely_oversold(&self) -> bool {
        self.current_cci
            .map_or(false, |cci| cci <= self.config.extreme_oversold_threshold)
    }

    /// Check if CCI is above zero line (bullish territory)
    pub fn is_bullish(&self) -> bool {
        self.current_cci.map_or(false, |cci| cci > Decimal::ZERO)
    }

    /// Check if CCI is below zero line (bearish territory)
    pub fn is_bearish(&self) -> bool {
        self.current_cci.map_or(false, |cci| cci < Decimal::ZERO)
    }

    /// Check if CCI is trending up
    pub fn is_trending_up(&self) -> bool {
        if let (Some(current), Some(previous)) = (self.current_cci, self.previous_cci) {
            current > previous
        } else {
            false
        }
    }

    /// Check if CCI is trending down
    pub fn is_trending_down(&self) -> bool {
        if let (Some(current), Some(previous)) = (self.current_cci, self.previous_cci) {
            current < previous
        } else {
            false
        }
    }

    /// Get CCI configuration
    pub fn config(&self) -> &CciConfig {
        &self.config
    }

    /// Reset the indicator
    pub fn reset(&mut self) {
        self.typical_prices.clear();
        self.current_cci = None;
        self.previous_cci = None;
    }

    /// Check if indicator is ready
    pub fn is_ready(&self) -> bool {
        self.current_cci.is_some()
    }

    /// Get minimum data points required
    pub fn min_data_points(&self) -> usize {
        self.config.period
    }

    /// Get data history length
    pub fn history_len(&self) -> usize {
        self.typical_prices.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;

    fn create_test_ohlc(
        high: Decimal,
        low: Decimal,
        close: Decimal,
        timestamp: DateTime<Utc>,
    ) -> OhlcData {
        OhlcData::new(
            (high + low) / dec!(2), // open as average
            high,
            low,
            close,
            dec!(1000),
            timestamp,
        )
        .unwrap()
    }

    #[test]
    fn test_cci_creation() {
        let config = CciConfig::default();
        let cci = CommodityChannelIndex::new(config).unwrap();
        assert_eq!(cci.config.period, 20);
        assert_eq!(cci.config.overbought_threshold, dec!(100));
        assert_eq!(cci.config.oversold_threshold, dec!(-100));
        assert!(!cci.is_ready());
    }

    #[test]
    fn test_cci_invalid_config() {
        let config = CciConfig {
            period: 0,
            ..Default::default()
        };
        assert!(CommodityChannelIndex::new(config).is_err());

        let config = CciConfig {
            overbought_threshold: dec!(-50),
            oversold_threshold: dec!(50),
            ..Default::default()
        };
        assert!(CommodityChannelIndex::new(config).is_err());

        let config = CciConfig {
            constant_factor: dec!(0),
            ..Default::default()
        };
        assert!(CommodityChannelIndex::new(config).is_err());
    }

    #[test]
    fn test_cci_calculation_uptrend() {
        let mut cci = CommodityChannelIndex::default().unwrap();
        let base_time = Utc::now();

        // Create uptrend data
        for i in 0..25 {
            let base_price = dec!(100) + Decimal::from(i);
            let ohlc = create_test_ohlc(
                base_price + dec!(2), // high
                base_price - dec!(1), // low
                base_price + dec!(1), // close (uptrend)
                base_time + chrono::Duration::seconds(i),
            );
            cci.update(&ohlc).unwrap();
        }

        assert!(cci.is_ready());

        if let Some(cci_value) = cci.current_value() {
            // In strong uptrend, CCI should be positive
            assert!(cci_value > Decimal::ZERO);
        }
    }

    #[test]
    fn test_cci_calculation_downtrend() {
        let mut cci = CommodityChannelIndex::default().unwrap();
        let base_time = Utc::now();

        // Create downtrend data
        for i in 0..25 {
            let base_price = dec!(100) - Decimal::from(i);
            let ohlc = create_test_ohlc(
                base_price + dec!(1), // high
                base_price - dec!(2), // low
                base_price - dec!(1), // close (downtrend)
                base_time + chrono::Duration::seconds(i),
            );
            cci.update(&ohlc).unwrap();
        }

        assert!(cci.is_ready());

        if let Some(cci_value) = cci.current_value() {
            // In strong downtrend, CCI should be negative
            assert!(cci_value < Decimal::ZERO);
        }
    }

    #[test]
    fn test_cci_extreme_levels() {
        let mut cci = CommodityChannelIndex::default().unwrap();
        let base_time = Utc::now();

        // Create extreme upward movement
        for i in 0..25 {
            let base_price = dec!(100) + Decimal::from(i * 5); // Aggressive uptrend
            let ohlc = create_test_ohlc(
                base_price + dec!(10), // high
                base_price - dec!(2),  // low
                base_price + dec!(8),  // close
                base_time + chrono::Duration::seconds(i),
            );
            cci.update(&ohlc).unwrap();
        }

        assert!(cci.is_ready());

        if let Some(cci_value) = cci.current_value() {
            // Should show high positive CCI
            assert!(cci_value > dec!(50)); // Should be significantly positive
        }
    }

    #[test]
    fn test_cci_signal_generation() {
        let mut cci = CommodityChannelIndex::default().unwrap();
        let base_time = Utc::now();

        // Generate data that crosses thresholds
        for i in 0..30 {
            let base_price = dec!(100);
            let volatility = if i < 15 { dec!(1) } else { dec!(10) }; // Increase volatility

            let ohlc = create_test_ohlc(
                base_price + volatility,
                base_price - volatility,
                base_price + (volatility * dec!(0.5)),
                base_time + chrono::Duration::seconds(i),
            );
            cci.update(&ohlc).unwrap();
        }

        assert!(cci.is_ready());

        // Try to generate a signal
        let _signal = cci.generate_signal(base_time).unwrap();
        // Signal generation depends on specific price movements
    }

    #[test]
    fn test_cci_level_checks() {
        let config = CciConfig {
            overbought_threshold: dec!(50),
            oversold_threshold: dec!(-50),
            extreme_overbought_threshold: dec!(100),
            extreme_oversold_threshold: dec!(-100),
            ..Default::default()
        };

        let mut cci = CommodityChannelIndex::new(config).unwrap();

        // Simulate overbought condition
        cci.current_cci = Some(dec!(75));
        assert!(cci.is_overbought());
        assert!(!cci.is_extremely_overbought());
        assert!(cci.is_bullish());

        // Simulate extreme overbought condition
        cci.current_cci = Some(dec!(150));
        assert!(cci.is_extremely_overbought());

        // Simulate oversold condition
        cci.current_cci = Some(dec!(-75));
        assert!(cci.is_oversold());
        assert!(!cci.is_extremely_oversold());
        assert!(cci.is_bearish());

        // Simulate extreme oversold condition
        cci.current_cci = Some(dec!(-150));
        assert!(cci.is_extremely_oversold());
    }

    #[test]
    fn test_cci_trend_detection() {
        let mut cci = CommodityChannelIndex::default().unwrap();

        cci.previous_cci = Some(dec!(50));
        cci.current_cci = Some(dec!(75));
        assert!(cci.is_trending_up());
        assert!(!cci.is_trending_down());

        cci.previous_cci = Some(dec!(75));
        cci.current_cci = Some(dec!(50));
        assert!(!cci.is_trending_up());
        assert!(cci.is_trending_down());
    }

    #[test]
    fn test_cci_reset() {
        let mut cci = CommodityChannelIndex::default().unwrap();
        let base_time = Utc::now();

        // Add some data
        for i in 0..15 {
            let ohlc = create_test_ohlc(
                dec!(101),
                dec!(99),
                dec!(100),
                base_time + chrono::Duration::seconds(i),
            );
            cci.update(&ohlc).unwrap();
        }

        assert!(cci.history_len() > 0);

        cci.reset();

        assert_eq!(cci.history_len(), 0);
        assert!(!cci.is_ready());
        assert!(cci.current_value().is_none());
    }

    #[test]
    fn test_cci_min_data_points() {
        let cci = CommodityChannelIndex::default().unwrap();
        assert_eq!(cci.min_data_points(), 20);
    }
}
