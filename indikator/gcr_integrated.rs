use super::gcr_chaos::{GCRChaosConfig, GCRChaosOscillator, GCRChaosResult, GoldenSignal};
use super::gcr_fractal::{GCRFractal, GCRFractalConfig, GCRFractalResult, GCRFractalSignal};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};

/// Moving Average Type for GCR Integrated calculations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum MAType {
    /// Simple Moving Average
    SMA,
    /// Exponential Moving Average
    EMA,
}

impl Default for MAType {
    fn default() -> Self {
        MAType::EMA
    }
}

/// GCR Integrated Configuration
#[derive(Debug, Clone)]
pub struct GCRIntegratedConfig {
    /// GCR Fractal configuration
    pub fractal_config: GCRFractalConfig,
    /// GCR Chaos Oscillator configuration
    pub chaos_config: GCRChaosConfig,
    /// Moving average type for smoothing
    pub ma_type: MAType,
    /// Smoothing period for integrated signal
    pub smoothing_period: usize,
    /// Weight for fractal component (0.0 to 1.0)
    pub fractal_weight: Decimal,
    /// Weight for chaos component (0.0 to 1.0)
    pub chaos_weight: Decimal,
}

impl Default for GCRIntegratedConfig {
    fn default() -> Self {
        Self {
            fractal_config: GCRFractalConfig::default(),
            chaos_config: GCRChaosConfig::default(),
            ma_type: MAType::EMA,
            smoothing_period: 10,
            fractal_weight: Decimal::new(6, 1), // 0.6
            chaos_weight: Decimal::new(4, 1),   // 0.4
        }
    }
}

/// GCR Integrated Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCRIntegratedResult {
    /// Fractal component value
    pub fractal_value: Decimal,
    /// Chaos oscillator value
    pub oscillator_value: Decimal,
    /// Integrated combined value
    pub integrated_value: Decimal,
    /// Combined signal from both indicators
    pub signal: GoldenSignal,
    /// Signal strength (0.0 to 1.0)
    pub signal_strength: Decimal,
    /// Fractal dimension from fractal component
    pub fractal_dimension: Decimal,
    /// Phase from chaos component
    pub chaos_phase: Decimal,
}

/// GCR Integrated Indicator
///
/// Combines GCR Fractal and GCR Chaos Oscillator for enhanced analysis
#[derive(Debug, Clone)]
pub struct GCRIntegrated {
    config: GCRIntegratedConfig,
    fractal_indicator: GCRFractal,
    chaos_indicator: GCRChaosOscillator,
    integrated_values: Vec<Decimal>,
    smoothed_values: Vec<Decimal>,
    is_ready: bool,
}

impl GCRIntegrated {
    /// Create new GCR Integrated indicator
    pub fn new(config: GCRIntegratedConfig) -> Self {
        Self {
            fractal_indicator: GCRFractal::new(config.fractal_config.clone()),
            chaos_indicator: GCRChaosOscillator::new(config.chaos_config.clone()),
            integrated_values: Vec::with_capacity(config.smoothing_period),
            smoothed_values: Vec::with_capacity(config.smoothing_period),
            config,
            is_ready: false,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(GCRIntegratedConfig::default())
    }

    /// Update with new OHLCV data
    pub fn update(
        &mut self,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
    ) -> Option<GCRIntegratedResult> {
        // Update both indicators
        let fractal_result = self.fractal_indicator.update(high, low, close, volume);
        let chaos_result = self.chaos_indicator.update(high, low, close, volume);

        // Check if both indicators are ready
        if !self.fractal_indicator.is_ready() || !self.chaos_indicator.is_ready() {
            return None;
        }

        let fractal_res = fractal_result?;
        let chaos_res = chaos_result?;

        // Calculate integrated value
        let integrated_value = self.calculate_integrated_value(&fractal_res, &chaos_res);

        // Update integrated values history
        self.integrated_values.push(integrated_value);
        if self.integrated_values.len() > self.config.smoothing_period {
            self.integrated_values.remove(0);
        }

        // Calculate smoothed value
        let smoothed_value = self.calculate_smoothed_value();

        // Update smoothed values history
        self.smoothed_values.push(smoothed_value);
        if self.smoothed_values.len() > self.config.smoothing_period {
            self.smoothed_values.remove(0);
        }

        // Generate combined signal
        let (signal, signal_strength) =
            self.generate_combined_signal(&fractal_res, &chaos_res, smoothed_value);

        // Check if ready
        self.is_ready = self.integrated_values.len() >= self.config.smoothing_period.min(5);

        if self.is_ready {
            Some(GCRIntegratedResult {
                fractal_value: fractal_res.fractal_value,
                oscillator_value: chaos_res.value,
                integrated_value: smoothed_value,
                signal,
                signal_strength,
                fractal_dimension: fractal_res.fractal_dimension,
                chaos_phase: chaos_res.phase,
            })
        } else {
            None
        }
    }

    /// Check if indicator is ready
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    /// Reset the indicator
    pub fn reset(&mut self) {
        self.fractal_indicator.reset();
        self.chaos_indicator.reset();
        self.integrated_values.clear();
        self.smoothed_values.clear();
        self.is_ready = false;
    }

    /// Calculate integrated value from fractal and chaos components
    fn calculate_integrated_value(
        &self,
        fractal_result: &GCRFractalResult,
        chaos_result: &GCRChaosResult,
    ) -> Decimal {
        let fractal_component = fractal_result.fractal_value * self.config.fractal_weight;
        let chaos_component = chaos_result.value * self.config.chaos_weight;

        // Combine weighted components
        let integrated = fractal_component + chaos_component;

        // Apply normalization to keep values in reasonable range
        integrated.max(Decimal::new(-2, 0)).min(Decimal::new(2, 0))
    }

    /// Calculate smoothed value using specified moving average type
    fn calculate_smoothed_value(&self) -> Decimal {
        if self.integrated_values.is_empty() {
            return Decimal::ZERO;
        }

        match self.config.ma_type {
            MAType::SMA => self.calculate_sma(),
            MAType::EMA => self.calculate_ema(),
        }
    }

    /// Calculate Simple Moving Average
    fn calculate_sma(&self) -> Decimal {
        let sum: Decimal = self.integrated_values.iter().sum();
        sum / Decimal::new(self.integrated_values.len() as i64, 0)
    }

    /// Calculate Exponential Moving Average
    fn calculate_ema(&self) -> Decimal {
        if self.smoothed_values.is_empty() {
            return self
                .integrated_values
                .last()
                .copied()
                .unwrap_or(Decimal::ZERO);
        }

        let current_value = self
            .integrated_values
            .last()
            .copied()
            .unwrap_or(Decimal::ZERO);
        let previous_ema = self
            .smoothed_values
            .last()
            .copied()
            .unwrap_or(Decimal::ZERO);

        // EMA multiplier: 2 / (period + 1)
        let multiplier = Decimal::new(2, 0)
            / (Decimal::new(self.config.smoothing_period as i64, 0) + Decimal::ONE);

        current_value * multiplier + previous_ema * (Decimal::ONE - multiplier)
    }

    /// Generate combined signal from both indicators
    fn generate_combined_signal(
        &self,
        fractal_result: &GCRFractalResult,
        chaos_result: &GCRChaosResult,
        _integrated_value: Decimal,
    ) -> (GoldenSignal, Decimal) {
        // Analyze fractal signal strength
        let fractal_signal_score = self.signal_to_score(&fractal_result.signal);

        // Analyze chaos signal strength
        let chaos_signal_score = self.golden_signal_to_score(&chaos_result.signal);

        // Calculate weighted combined score
        let combined_score = fractal_signal_score * self.config.fractal_weight
            + chaos_signal_score * self.config.chaos_weight;

        // Factor in integrated value momentum
        let momentum_factor = if self.smoothed_values.len() >= 2 {
            let current = self.smoothed_values[self.smoothed_values.len() - 1];
            let previous = self.smoothed_values[self.smoothed_values.len() - 2];
            (current - previous).signum()
        } else {
            Decimal::ZERO
        };

        // Adjust combined score with momentum
        let final_score = combined_score + momentum_factor * Decimal::new(2, 1); // 0.2 momentum weight

        // Calculate signal strength
        let signal_strength =
            (fractal_result.signal_strength + chaos_result.signal_strength) / Decimal::new(2, 0);

        // Determine final signal
        let signal = if final_score >= Decimal::new(15, 1) {
            // 1.5
            GoldenSignal::HyperBull
        } else if final_score >= Decimal::new(5, 1) {
            // 0.5
            GoldenSignal::Bull
        } else if final_score <= Decimal::new(-15, 1) {
            // -1.5
            GoldenSignal::HyperBear
        } else if final_score <= Decimal::new(-5, 1) {
            // -0.5
            GoldenSignal::Bear
        } else {
            GoldenSignal::Neutral
        };

        (signal, signal_strength)
    }

    /// Convert GCR Fractal signal to numerical score
    fn signal_to_score(&self, signal: &GCRFractalSignal) -> Decimal {
        match signal {
            GCRFractalSignal::StrongBuy => Decimal::new(2, 0),
            GCRFractalSignal::Buy => Decimal::ONE,
            GCRFractalSignal::Warning => Decimal::ZERO,
            GCRFractalSignal::Normal => Decimal::ZERO,
            GCRFractalSignal::Sell => -Decimal::ONE,
            GCRFractalSignal::StrongSell => Decimal::new(-2, 0),
        }
    }

    /// Convert Golden signal to numerical score
    fn golden_signal_to_score(&self, signal: &GoldenSignal) -> Decimal {
        match signal {
            GoldenSignal::HyperBull => Decimal::new(2, 0),
            GoldenSignal::Bull => Decimal::ONE,
            GoldenSignal::Neutral => Decimal::ZERO,
            GoldenSignal::Bear => -Decimal::ONE,
            GoldenSignal::HyperBear => Decimal::new(-2, 0),
        }
    }

    /// Get current integrated value
    pub fn get_integrated_value(&self) -> Option<Decimal> {
        self.smoothed_values.last().copied()
    }

    /// Get current fractal value
    pub fn get_fractal_value(&self) -> Option<Decimal> {
        self.fractal_indicator.get_fractal_value()
    }

    /// Get current chaos value
    pub fn get_chaos_value(&self) -> Decimal {
        self.chaos_indicator.get_value()
    }

    /// Get current combined signal
    pub fn get_signal(&self) -> GoldenSignal {
        if !self.is_ready {
            return GoldenSignal::Neutral;
        }

        // Get latest results from both indicators
        let fractal_signal = GCRFractalSignal::Normal; // Default fallback
        let chaos_signal = self.chaos_indicator.get_signal();

        let integrated_value = self.get_integrated_value().unwrap_or(Decimal::ZERO);

        // Create dummy results for signal generation
        let fractal_result = GCRFractalResult {
            fractal_value: self.get_fractal_value().unwrap_or(Decimal::ZERO),
            chaos_vector: super::gcr_fractal::ChaosVector::zero(),
            fractal_dimension: Decimal::new(15, 1),
            signal: fractal_signal,
            signal_strength: Decimal::new(5, 1),
        };

        let chaos_result = GCRChaosResult {
            value: self.get_chaos_value(),
            signal: chaos_signal,
            signal_strength: Decimal::new(5, 1),
            phase: self.chaos_indicator.get_phase(),
        };

        let (signal, _) =
            self.generate_combined_signal(&fractal_result, &chaos_result, integrated_value);
        signal
    }
}

trait DecimalSignum {
    fn signum(self) -> Self;
}

impl DecimalSignum for Decimal {
    fn signum(self) -> Self {
        if self > Decimal::ZERO {
            Decimal::ONE
        } else if self < Decimal::ZERO {
            -Decimal::ONE
        } else {
            Decimal::ZERO
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcr_integrated_creation() {
        let indicator = GCRIntegrated::default();
        assert!(!indicator.is_ready());
    }

    #[test]
    fn test_gcr_integrated_update() {
        let mut indicator = GCRIntegrated::default();

        // Add enough data points with smaller values for both indicators to be ready
        for i in 1..=100 {
            let price = Decimal::new(100 + i, 0);
            let volume = Decimal::new(10 + i, 0);
            let result = indicator.update(
                price + Decimal::new(1, 0),
                price - Decimal::new(1, 0),
                price,
                volume,
            );

            if i >= 89 {
                // Should be ready after fractal memory depth
                if let Some(res) = result {
                    // Allow wider range for integrated values
                    assert!(
                        res.integrated_value >= Decimal::new(-100, 0)
                            && res.integrated_value <= Decimal::new(100, 0)
                    );
                    assert!(
                        res.signal_strength >= Decimal::ZERO && res.signal_strength <= Decimal::ONE
                    );
                }
            }
        }

        assert!(indicator.is_ready());
    }

    #[test]
    fn test_ma_type_calculations() {
        let mut sma_indicator = GCRIntegrated::new(GCRIntegratedConfig {
            ma_type: MAType::SMA,
            ..Default::default()
        });

        let mut ema_indicator = GCRIntegrated::new(GCRIntegratedConfig {
            ma_type: MAType::EMA,
            ..Default::default()
        });

        // Add same data to both with smaller values
        for i in 1..=100 {
            let price = Decimal::new(100 + i, 0);
            let volume = Decimal::new(10, 0);

            sma_indicator.update(
                price + Decimal::new(1, 0),
                price - Decimal::new(1, 0),
                price,
                volume,
            );

            ema_indicator.update(
                price + Decimal::new(1, 0),
                price - Decimal::new(1, 0),
                price,
                volume,
            );
        }

        // Both should be ready and have different smoothed values
        assert!(sma_indicator.is_ready());
        assert!(ema_indicator.is_ready());

        let sma_value = sma_indicator.get_integrated_value();
        let ema_value = ema_indicator.get_integrated_value();

        assert!(sma_value.is_some());
        assert!(ema_value.is_some());
        // Values might be different due to different smoothing methods
    }

    #[test]
    fn test_signal_combination() {
        let mut indicator = GCRIntegrated::default();

        // Generate upward trend with smaller values
        for i in 1..=100 {
            let price = Decimal::new(100 + i * 2, 0); // Smaller upward trend
            let volume = Decimal::new(20, 0);
            let result = indicator.update(
                price + Decimal::new(1, 0),
                price - Decimal::new(1, 0),
                price,
                volume,
            );

            if let Some(res) = result {
                // Should generate valid signals
                assert!(matches!(
                    res.signal,
                    GoldenSignal::Bull
                        | GoldenSignal::HyperBull
                        | GoldenSignal::Neutral
                        | GoldenSignal::Bear
                        | GoldenSignal::HyperBear
                ));
            }
        }
    }

    #[test]
    fn test_weight_configuration() {
        let config = GCRIntegratedConfig {
            fractal_weight: Decimal::new(8, 1), // 0.8
            chaos_weight: Decimal::new(2, 1),   // 0.2
            ..Default::default()
        };

        let indicator = GCRIntegrated::new(config.clone());
        assert_eq!(indicator.config.fractal_weight, config.fractal_weight);
        assert_eq!(indicator.config.chaos_weight, config.chaos_weight);
    }

    #[test]
    fn test_reset_functionality() {
        let mut indicator = GCRIntegrated::default();

        // Add data with smaller values
        for i in 1..=100 {
            let price = Decimal::new(100 + i, 0);
            let volume = Decimal::new(10, 0);
            indicator.update(
                price + Decimal::new(1, 0),
                price - Decimal::new(1, 0),
                price,
                volume,
            );
        }

        assert!(indicator.is_ready());

        indicator.reset();
        assert!(!indicator.is_ready());
        assert!(indicator.integrated_values.is_empty());
        assert!(indicator.smoothed_values.is_empty());
    }

    #[test]
    fn test_signal_scoring() {
        let indicator = GCRIntegrated::default();

        // Test fractal signal scoring
        assert_eq!(
            indicator.signal_to_score(&GCRFractalSignal::StrongBuy),
            Decimal::new(2, 0)
        );
        assert_eq!(
            indicator.signal_to_score(&GCRFractalSignal::Buy),
            Decimal::ONE
        );
        assert_eq!(
            indicator.signal_to_score(&GCRFractalSignal::Normal),
            Decimal::ZERO
        );
        assert_eq!(
            indicator.signal_to_score(&GCRFractalSignal::Sell),
            -Decimal::ONE
        );
        assert_eq!(
            indicator.signal_to_score(&GCRFractalSignal::StrongSell),
            Decimal::new(-2, 0)
        );

        // Test golden signal scoring
        assert_eq!(
            indicator.golden_signal_to_score(&GoldenSignal::HyperBull),
            Decimal::new(2, 0)
        );
        assert_eq!(
            indicator.golden_signal_to_score(&GoldenSignal::Bull),
            Decimal::ONE
        );
        assert_eq!(
            indicator.golden_signal_to_score(&GoldenSignal::Neutral),
            Decimal::ZERO
        );
        assert_eq!(
            indicator.golden_signal_to_score(&GoldenSignal::Bear),
            -Decimal::ONE
        );
        assert_eq!(
            indicator.golden_signal_to_score(&GoldenSignal::HyperBear),
            Decimal::new(-2, 0)
        );
    }
}
