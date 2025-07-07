//! Stochastic Oscillator Implementation
//!
//! Stochastic Oscillator, momentum indikatörüdür ve fiyatın belirli bir dönemdeki
//! yüksek-düşük aralığı içindeki pozisyonunu ölçer.
//!
//! %K = ((Current Close - Lowest Low) / (Highest High - Lowest Low)) * 100
//! %D = SMA of %K (typically 3 periods)

use crate::types::{
    IndicatorResult, OhlcData, Signal, SignalData, SignalStrength, TechnicalAnalysisError,
};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::VecDeque;

/// Stochastic Configuration
#[derive(Debug, Clone)]
pub struct StochasticConfig {
    /// %K period (typically 14)
    pub k_period: usize,
    /// %D period - SMA of %K (typically 3)
    pub d_period: usize,
    /// Overbought threshold (typically 80)
    pub overbought_threshold: Decimal,
    /// Oversold threshold (typically 20)
    pub oversold_threshold: Decimal,
    /// Strong signal threshold (typically 90/10)
    pub strong_overbought_threshold: Decimal,
    /// Strong oversold threshold (typically 10)
    pub strong_oversold_threshold: Decimal,
}

impl Default for StochasticConfig {
    fn default() -> Self {
        Self {
            k_period: 14,
            d_period: 3,
            overbought_threshold: dec!(80),
            oversold_threshold: dec!(20),
            strong_overbought_threshold: dec!(90),
            strong_oversold_threshold: dec!(10),
        }
    }
}

/// Stochastic Oscillator Implementation
#[derive(Debug, Clone)]
pub struct StochasticOscillator {
    config: StochasticConfig,
    ohlc_history: VecDeque<OhlcData>,
    k_values: VecDeque<Decimal>,
    d_values: VecDeque<Decimal>,
    current_k: Option<Decimal>,
    current_d: Option<Decimal>,
}

impl StochasticOscillator {
    /// Create new Stochastic Oscillator
    pub fn new(config: StochasticConfig) -> Result<Self, TechnicalAnalysisError> {
        if config.k_period == 0 {
            return Err(TechnicalAnalysisError::invalid_config(format!(
                "K period must be greater than 0, got: {}",
                config.k_period
            )));
        }

        if config.d_period == 0 {
            return Err(TechnicalAnalysisError::invalid_config(format!(
                "D period must be greater than 0, got: {}",
                config.d_period
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
            k_values: VecDeque::with_capacity(100),
            d_values: VecDeque::with_capacity(100),
            current_k: None,
            current_d: None,
        })
    }

    /// Create Stochastic with default configuration
    pub fn default() -> Result<Self, TechnicalAnalysisError> {
        Self::new(StochasticConfig::default())
    }

    /// Update Stochastic with new OHLC data
    pub fn update(
        &mut self,
        ohlc: &OhlcData,
    ) -> Result<Option<IndicatorResult>, TechnicalAnalysisError> {
        // Add current OHLC to history
        self.ohlc_history.push_back(ohlc.clone());

        // Keep only necessary history
        if self.ohlc_history.len() > self.config.k_period {
            self.ohlc_history.pop_front();
        }

        // Calculate %K if we have enough data
        if self.ohlc_history.len() >= self.config.k_period {
            let k_value = self.calculate_k()?;
            self.current_k = Some(k_value);

            // Add %K value to history
            self.k_values.push_back(k_value);

            // Keep only necessary %K history for %D calculation
            if self.k_values.len() > self.config.d_period {
                self.k_values.pop_front();
            }

            // Calculate %D if we have enough %K values
            if self.k_values.len() >= self.config.d_period {
                let d_value = self.calculate_d()?;
                self.current_d = Some(d_value);

                // Add %D value to history
                self.d_values.push_back(d_value);

                // Keep limited %D history
                if self.d_values.len() > 50 {
                    self.d_values.pop_front();
                }

                // Return %D as the main indicator result
                return Ok(Some(IndicatorResult::new(
                    d_value,
                    ohlc.timestamp,
                    "Stochastic".to_string(),
                )));
            } else {
                // Return %K if %D not ready yet
                return Ok(Some(IndicatorResult::new(
                    k_value,
                    ohlc.timestamp,
                    "Stochastic_K".to_string(),
                )));
            }
        }

        Ok(None)
    }

    /// Calculate %K value
    fn calculate_k(&self) -> Result<Decimal, TechnicalAnalysisError> {
        if self.ohlc_history.len() < self.config.k_period {
            return Err(TechnicalAnalysisError::insufficient_data(
                self.config.k_period,
                self.ohlc_history.len(),
            ));
        }

        // Find highest high and lowest low in the period
        let mut highest_high = self.ohlc_history[0].high;
        let mut lowest_low = self.ohlc_history[0].low;

        for ohlc in &self.ohlc_history {
            if ohlc.high > highest_high {
                highest_high = ohlc.high;
            }
            if ohlc.low < lowest_low {
                lowest_low = ohlc.low;
            }
        }

        let current_close = self.ohlc_history.back().unwrap().close;

        // Calculate %K
        let range = highest_high - lowest_low;
        if range == Decimal::ZERO {
            // If no price movement, return 50 (middle)
            Ok(dec!(50))
        } else {
            let k = ((current_close - lowest_low) / range) * dec!(100);
            Ok(k.max(Decimal::ZERO).min(dec!(100)))
        }
    }

    /// Calculate %D value (SMA of %K)
    fn calculate_d(&self) -> Result<Decimal, TechnicalAnalysisError> {
        if self.k_values.len() < self.config.d_period {
            return Err(TechnicalAnalysisError::insufficient_data(
                self.config.d_period,
                self.k_values.len(),
            ));
        }

        // Calculate simple moving average of %K values
        let sum: Decimal = self.k_values.iter().take(self.config.d_period).sum();
        Ok(sum / Decimal::from(self.config.d_period))
    }

    /// Generate trading signal based on Stochastic values
    pub fn generate_signal(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        if let (Some(k_value), Some(d_value)) = (self.current_k, self.current_d) {
            // Primary signal based on %D value
            let (primary_signal, primary_strength) =
                if d_value >= self.config.strong_overbought_threshold {
                    (Signal::Sell, SignalStrength::Strong)
                } else if d_value >= self.config.overbought_threshold {
                    (Signal::Sell, SignalStrength::Weak)
                } else if d_value <= self.config.strong_oversold_threshold {
                    (Signal::Buy, SignalStrength::Strong)
                } else if d_value <= self.config.oversold_threshold {
                    (Signal::Buy, SignalStrength::Weak)
                } else {
                    return Ok(None); // No signal in neutral zone
                };

            // Enhance signal strength based on %K and %D crossover
            let enhanced_strength =
                self.enhance_signal_strength(k_value, d_value, primary_strength);

            // Calculate confidence
            let confidence = self.calculate_confidence(k_value, d_value);

            Ok(Some(SignalData::new(
                primary_signal,
                enhanced_strength,
                confidence,
                timestamp,
                "Stochastic".to_string(),
            )?))
        } else {
            Ok(None)
        }
    }

    /// Enhance signal strength based on %K and %D relationship
    fn enhance_signal_strength(
        &self,
        k_value: Decimal,
        d_value: Decimal,
        base_strength: SignalStrength,
    ) -> SignalStrength {
        // Check if %K and %D are both in the same zone (confirming signal)
        let k_overbought = k_value >= self.config.overbought_threshold;
        let d_overbought = d_value >= self.config.overbought_threshold;
        let k_oversold = k_value <= self.config.oversold_threshold;
        let d_oversold = d_value <= self.config.oversold_threshold;

        // Check for crossover conditions
        let bullish_crossover = k_value > d_value && (k_oversold || d_oversold);
        let bearish_crossover = k_value < d_value && (k_overbought || d_overbought);

        match base_strength {
            SignalStrength::VeryWeak => {
                if (bullish_crossover || bearish_crossover)
                    || (k_overbought && d_overbought)
                    || (k_oversold && d_oversold)
                {
                    SignalStrength::Weak
                } else {
                    SignalStrength::VeryWeak
                }
            },
            SignalStrength::Weak => {
                if (bullish_crossover || bearish_crossover)
                    || (k_overbought && d_overbought)
                    || (k_oversold && d_oversold)
                {
                    SignalStrength::Strong
                } else {
                    SignalStrength::Weak
                }
            },
            SignalStrength::Moderate => {
                if (bullish_crossover || bearish_crossover)
                    || (k_overbought && d_overbought)
                    || (k_oversold && d_oversold)
                {
                    SignalStrength::Strong
                } else {
                    SignalStrength::Moderate
                }
            },
            SignalStrength::Strong => {
                // Keep strong if conditions are met, otherwise downgrade
                if (k_overbought && d_overbought) || (k_oversold && d_oversold) {
                    SignalStrength::Strong
                } else {
                    SignalStrength::Moderate
                }
            },
            SignalStrength::VeryStrong => {
                // Keep very strong if conditions are met, otherwise downgrade
                if (k_overbought && d_overbought) || (k_oversold && d_oversold) {
                    SignalStrength::VeryStrong
                } else {
                    SignalStrength::Strong
                }
            },
        }
    }

    /// Calculate signal confidence
    fn calculate_confidence(&self, k_value: Decimal, d_value: Decimal) -> Decimal {
        // Base confidence on how extreme the values are
        let k_confidence = if k_value >= self.config.overbought_threshold {
            let range = dec!(100) - self.config.overbought_threshold;
            (k_value - self.config.overbought_threshold) / range
        } else if k_value <= self.config.oversold_threshold {
            let range = self.config.oversold_threshold;
            (self.config.oversold_threshold - k_value) / range
        } else {
            dec!(0.3) // Low confidence in neutral zone
        };

        let d_confidence = if d_value >= self.config.overbought_threshold {
            let range = dec!(100) - self.config.overbought_threshold;
            (d_value - self.config.overbought_threshold) / range
        } else if d_value <= self.config.oversold_threshold {
            let range = self.config.oversold_threshold;
            (self.config.oversold_threshold - d_value) / range
        } else {
            dec!(0.3) // Low confidence in neutral zone
        };

        // Average the confidences and ensure it's between 0 and 1
        ((k_confidence + d_confidence) / dec!(2))
            .max(dec!(0.1))
            .min(Decimal::ONE)
    }

    /// Get current %K value
    pub fn current_k(&self) -> Option<Decimal> {
        self.current_k
    }

    /// Get current %D value
    pub fn current_d(&self) -> Option<Decimal> {
        self.current_d
    }

    /// Check if Stochastic is in overbought condition
    pub fn is_overbought(&self) -> bool {
        self.current_d
            .map_or(false, |d| d >= self.config.overbought_threshold)
    }

    /// Check if Stochastic is in oversold condition
    pub fn is_oversold(&self) -> bool {
        self.current_d
            .map_or(false, |d| d <= self.config.oversold_threshold)
    }

    /// Check for bullish crossover (%K crosses above %D)
    pub fn is_bullish_crossover(&self) -> bool {
        if let (Some(k), Some(d)) = (self.current_k, self.current_d) {
            k > d && self.is_oversold()
        } else {
            false
        }
    }

    /// Check for bearish crossover (%K crosses below %D)
    pub fn is_bearish_crossover(&self) -> bool {
        if let (Some(k), Some(d)) = (self.current_k, self.current_d) {
            k < d && self.is_overbought()
        } else {
            false
        }
    }

    /// Get Stochastic configuration
    pub fn config(&self) -> &StochasticConfig {
        &self.config
    }

    /// Reset the indicator
    pub fn reset(&mut self) {
        self.ohlc_history.clear();
        self.k_values.clear();
        self.d_values.clear();
        self.current_k = None;
        self.current_d = None;
    }

    /// Check if indicator is ready (has enough data for %D)
    pub fn is_ready(&self) -> bool {
        self.current_d.is_some()
    }

    /// Check if %K is ready (has enough data for %K)
    pub fn is_k_ready(&self) -> bool {
        self.current_k.is_some()
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
        high: Decimal,
        low: Decimal,
        close: Decimal,
        timestamp: DateTime<Utc>,
    ) -> OhlcData {
        OhlcData::new(close, high, low, close, dec!(1000), timestamp).unwrap()
    }

    #[test]
    fn test_stochastic_creation() {
        let config = StochasticConfig::default();
        let stoch = StochasticOscillator::new(config).unwrap();
        assert_eq!(stoch.config.k_period, 14);
        assert_eq!(stoch.config.d_period, 3);
        assert!(!stoch.is_ready());
    }

    #[test]
    fn test_stochastic_invalid_periods() {
        let config = StochasticConfig {
            k_period: 0,
            ..Default::default()
        };
        assert!(StochasticOscillator::new(config).is_err());

        let config = StochasticConfig {
            d_period: 0,
            ..Default::default()
        };
        assert!(StochasticOscillator::new(config).is_err());
    }

    #[test]
    fn test_stochastic_k_calculation() {
        let mut stoch = StochasticOscillator::default().unwrap();
        let base_time = Utc::now();

        // Create test data where close is in the middle of high-low range
        // This should result in %K around 50
        for i in 0..14 {
            let ohlc = create_test_ohlc(
                dec!(110), // high
                dec!(90),  // low
                dec!(100), // close (middle)
                base_time + chrono::Duration::seconds(i),
            );
            stoch.update(&ohlc).unwrap();
        }

        assert!(stoch.is_k_ready());
        let k_value = stoch.current_k().unwrap();
        assert!(k_value >= dec!(45) && k_value <= dec!(55)); // Should be around 50
    }

    #[test]
    fn test_stochastic_extreme_values() {
        let mut stoch = StochasticOscillator::default().unwrap();
        let base_time = Utc::now();

        // Test %K = 100 (close at high)
        for i in 0..14 {
            let ohlc = create_test_ohlc(
                dec!(110), // high
                dec!(90),  // low
                dec!(110), // close at high
                base_time + chrono::Duration::seconds(i),
            );
            stoch.update(&ohlc).unwrap();
        }

        let k_value = stoch.current_k().unwrap();
        assert_eq!(k_value, dec!(100));

        // Reset and test %K = 0 (close at low)
        stoch.reset();
        for i in 0..14 {
            let ohlc = create_test_ohlc(
                dec!(110), // high
                dec!(90),  // low
                dec!(90),  // close at low
                base_time + chrono::Duration::seconds(i),
            );
            stoch.update(&ohlc).unwrap();
        }

        let k_value = stoch.current_k().unwrap();
        assert_eq!(k_value, dec!(0));
    }

    #[test]
    fn test_stochastic_d_calculation() {
        let mut stoch = StochasticOscillator::default().unwrap();
        let base_time = Utc::now();

        // Add enough data to calculate %D
        for i in 0..17 {
            // 14 for %K + 3 more for %D
            let ohlc = create_test_ohlc(
                dec!(110),
                dec!(90),
                dec!(100),
                base_time + chrono::Duration::seconds(i),
            );
            stoch.update(&ohlc).unwrap();
        }

        assert!(stoch.is_ready());
        assert!(stoch.current_d().is_some());
    }

    #[test]
    fn test_stochastic_overbought_signal() {
        let mut stoch = StochasticOscillator::default().unwrap();
        let base_time = Utc::now();

        // Create data that will result in overbought condition
        for i in 0..17 {
            let ohlc = create_test_ohlc(
                dec!(110),
                dec!(90),
                dec!(109), // Close near high to get high %K values
                base_time + chrono::Duration::seconds(i),
            );
            stoch.update(&ohlc).unwrap();
        }

        assert!(stoch.is_overbought());

        let signal = stoch.generate_signal(base_time).unwrap();
        assert!(signal.is_some());
        let signal = signal.unwrap();
        assert_eq!(signal.signal, Signal::Sell);
    }

    #[test]
    fn test_stochastic_oversold_signal() {
        let mut stoch = StochasticOscillator::default().unwrap();
        let base_time = Utc::now();

        // Create data that will result in oversold condition
        for i in 0..17 {
            let ohlc = create_test_ohlc(
                dec!(110),
                dec!(90),
                dec!(91), // Close near low to get low %K values
                base_time + chrono::Duration::seconds(i),
            );
            stoch.update(&ohlc).unwrap();
        }

        assert!(stoch.is_oversold());

        let signal = stoch.generate_signal(base_time).unwrap();
        assert!(signal.is_some());
        let signal = signal.unwrap();
        assert_eq!(signal.signal, Signal::Buy);
    }

    #[test]
    fn test_stochastic_crossover_detection() {
        let mut stoch = StochasticOscillator::default().unwrap();
        let base_time = Utc::now();

        // First create oversold condition
        for i in 0..17 {
            let ohlc = create_test_ohlc(
                dec!(110),
                dec!(90),
                dec!(91),
                base_time + chrono::Duration::seconds(i),
            );
            stoch.update(&ohlc).unwrap();
        }

        // Then simulate price recovery (should create bullish crossover)
        for i in 17..20 {
            let ohlc = create_test_ohlc(
                dec!(110),
                dec!(90),
                dec!(105), // Higher close
                base_time + chrono::Duration::seconds(i),
            );
            stoch.update(&ohlc).unwrap();
        }

        // Note: Crossover detection requires comparing with previous values
        // This is a simplified test
        assert!(stoch.current_k().unwrap() > dec!(50));
    }

    #[test]
    fn test_stochastic_reset() {
        let mut stoch = StochasticOscillator::default().unwrap();
        let base_time = Utc::now();

        // Add some data
        for i in 0..10 {
            let ohlc = create_test_ohlc(
                dec!(110),
                dec!(90),
                dec!(100),
                base_time + chrono::Duration::seconds(i),
            );
            stoch.update(&ohlc).unwrap();
        }

        assert!(stoch.history_len() > 0);

        stoch.reset();

        assert_eq!(stoch.history_len(), 0);
        assert!(!stoch.is_ready());
        assert!(stoch.current_k().is_none());
        assert!(stoch.current_d().is_none());
    }
}
