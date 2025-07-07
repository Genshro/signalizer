use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// GCR Chaos Oscillator Signal Types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GoldenSignal {
    /// Very strong bullish signal
    HyperBull,
    /// Bullish signal
    Bull,
    /// Neutral signal
    Neutral,
    /// Bearish signal
    Bear,
    /// Very strong bearish signal
    HyperBear,
}

impl Default for GoldenSignal {
    fn default() -> Self {
        GoldenSignal::Neutral
    }
}

/// GCR Chaos Oscillator Configuration
#[derive(Debug, Clone)]
pub struct GCRChaosConfig {
    /// Base frequency value (default: 0.618)
    pub base_frequency: Decimal,
    /// Fibonacci ratio (Golden ratio by default)
    pub phi: Decimal,
    /// Lookback period for calculations
    pub lookback_period: usize,
    /// Sensitivity for signal generation
    pub sensitivity: Decimal,
}

impl Default for GCRChaosConfig {
    fn default() -> Self {
        Self {
            base_frequency: Decimal::new(618, 3), // 0.618
            phi: Decimal::new(1618, 3),           // Golden ratio ≈ 1.618
            lookback_period: 14,
            sensitivity: Decimal::ONE,
        }
    }
}

/// GCR Chaos Oscillator Result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GCRChaosResult {
    /// Oscillator value
    pub value: Decimal,
    /// Generated signal
    pub signal: GoldenSignal,
    /// Signal strength (0.0 to 1.0)
    pub signal_strength: Decimal,
    /// Phase angle
    pub phase: Decimal,
}

/// GCR Chaos Oscillator Indicator
#[derive(Debug, Clone)]
pub struct GCRChaosOscillator {
    config: GCRChaosConfig,
    price_history: VecDeque<Decimal>,
    high_history: VecDeque<Decimal>,
    low_history: VecDeque<Decimal>,
    volume_history: VecDeque<Decimal>,
    phase: Decimal,
    previous_value: Decimal,
    is_ready: bool,
}

impl GCRChaosOscillator {
    /// Create new GCR Chaos Oscillator indicator
    pub fn new(config: GCRChaosConfig) -> Self {
        Self {
            price_history: VecDeque::with_capacity(config.lookback_period),
            high_history: VecDeque::with_capacity(config.lookback_period),
            low_history: VecDeque::with_capacity(config.lookback_period),
            volume_history: VecDeque::with_capacity(config.lookback_period),
            config,
            phase: Decimal::ZERO,
            previous_value: Decimal::ZERO,
            is_ready: false,
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(GCRChaosConfig::default())
    }

    /// Update with new OHLCV data
    pub fn update(
        &mut self,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
    ) -> Option<GCRChaosResult> {
        // Update price histories
        self.price_history.push_back(close);
        self.high_history.push_back(high);
        self.low_history.push_back(low);
        self.volume_history.push_back(volume);

        // Maintain lookback period
        if self.price_history.len() > self.config.lookback_period {
            self.price_history.pop_front();
            self.high_history.pop_front();
            self.low_history.pop_front();
            self.volume_history.pop_front();
        }

        // Check if ready
        self.is_ready = self.price_history.len() >= self.config.lookback_period;

        if !self.is_ready {
            return None;
        }

        // Calculate fractal input
        let fractal_input = self.calculate_fractal_input();

        // Calculate oscillator value
        let (value, signal, signal_strength) = self.calculate_oscillator_value(fractal_input);

        // Update previous value
        self.previous_value = value;

        Some(GCRChaosResult {
            value,
            signal,
            signal_strength,
            phase: self.phase,
        })
    }

    /// Check if indicator is ready
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    /// Reset the indicator
    pub fn reset(&mut self) {
        self.price_history.clear();
        self.high_history.clear();
        self.low_history.clear();
        self.volume_history.clear();
        self.phase = Decimal::ZERO;
        self.previous_value = Decimal::ZERO;
        self.is_ready = false;
    }

    /// Calculate fractal input from price data
    fn calculate_fractal_input(&self) -> Decimal {
        if self.price_history.len() < self.config.lookback_period {
            return Decimal::ZERO;
        }

        let current_price = *self.price_history.back().unwrap();
        let lookback_price = self.price_history[0];

        // Calculate price change percentage
        let price_change = if lookback_price != Decimal::ZERO {
            (current_price - lookback_price) / lookback_price
        } else {
            Decimal::ZERO
        };

        // Calculate volatility over lookback period
        let mut volatility = Decimal::ZERO;
        for i in 0..self.config.lookback_period {
            if i < self.high_history.len() && i < self.low_history.len() {
                let high = self.high_history[i];
                let low = self.low_history[i];
                if low != Decimal::ZERO {
                    volatility += (high - low) / low;
                }
            }
        }
        volatility /= Decimal::new(self.config.lookback_period as i64, 0);

        // Combine price change and volatility for fractal input
        let fractal_input = price_change + volatility * Decimal::new(5, 1); // 0.5 weight for volatility

        fractal_input
    }

    /// Calculate oscillator value and signal
    fn calculate_oscillator_value(
        &mut self,
        fractal_input: Decimal,
    ) -> (Decimal, GoldenSignal, Decimal) {
        let base_frequency = self.config.base_frequency;
        let phi = self.config.phi;
        let sensitivity = self.config.sensitivity;

        // Update phase based on fractal input
        self.phase += fractal_input * base_frequency;

        // Calculate golden ratio wave function
        let phi_wave = (self.phase * phi).sin().unwrap_or(Decimal::ZERO);

        // Calculate chaos modulation
        let chaos_factor = self.calculate_chaos_modulation(fractal_input);

        // Combine components
        let raw_value = phi_wave * chaos_factor;

        // Apply smoothing with previous value
        let smoothing_factor = Decimal::new(7, 1); // 0.7
        let value =
            raw_value * (Decimal::ONE - smoothing_factor) + self.previous_value * smoothing_factor;

        // Generate signal
        let (signal, signal_strength) = self.generate_signal(value, fractal_input, sensitivity);

        (value, signal, signal_strength)
    }

    /// Calculate chaos modulation factor
    fn calculate_chaos_modulation(&self, fractal_input: Decimal) -> Decimal {
        // Use volume-weighted chaos calculation
        let mut volume_weighted_chaos = Decimal::ZERO;
        let mut total_volume = Decimal::ZERO;

        for (i, &volume) in self.volume_history.iter().enumerate() {
            if i < self.price_history.len() {
                let price = self.price_history[i];
                let chaos_component = (price * self.config.phi).sin().unwrap_or(Decimal::ZERO);
                volume_weighted_chaos += chaos_component * volume;
                total_volume += volume;
            }
        }

        let chaos_factor = if total_volume > Decimal::ZERO {
            volume_weighted_chaos / total_volume
        } else {
            Decimal::ZERO
        };

        // Apply fractal input modulation
        let modulation = Decimal::ONE + fractal_input * Decimal::new(2, 1); // 0.2 scaling
        chaos_factor * modulation
    }

    /// Generate trading signal based on oscillator value
    fn generate_signal(
        &self,
        value: Decimal,
        fractal_input: Decimal,
        sensitivity: Decimal,
    ) -> (GoldenSignal, Decimal) {
        // Calculate derivative (momentum)
        let derivative = value - self.previous_value;

        // Determine trend direction
        let is_uptrend = fractal_input > Decimal::ZERO;

        // Apply sensitivity scaling
        let scaled_derivative = derivative * sensitivity;
        let scaled_value = value * sensitivity;

        // Signal thresholds
        let hyper_threshold = Decimal::new(8, 1); // 0.8
        let normal_threshold = Decimal::new(3, 1); // 0.3

        // Calculate signal strength
        let signal_strength = scaled_value.abs().min(Decimal::ONE);

        // Determine signal
        let signal =
            if scaled_value > hyper_threshold && scaled_derivative > Decimal::ZERO && is_uptrend {
                GoldenSignal::HyperBull
            } else if scaled_value > normal_threshold && scaled_derivative >= Decimal::ZERO {
                GoldenSignal::Bull
            } else if scaled_value < -hyper_threshold
                && scaled_derivative < Decimal::ZERO
                && !is_uptrend
            {
                GoldenSignal::HyperBear
            } else if scaled_value < -normal_threshold && scaled_derivative <= Decimal::ZERO {
                GoldenSignal::Bear
            } else {
                GoldenSignal::Neutral
            };

        (signal, signal_strength)
    }

    /// Get current oscillator value
    pub fn get_value(&self) -> Decimal {
        self.previous_value
    }

    /// Get current phase
    pub fn get_phase(&self) -> Decimal {
        self.phase
    }

    /// Get current signal
    pub fn get_signal(&self) -> GoldenSignal {
        if !self.is_ready {
            return GoldenSignal::Neutral;
        }

        let fractal_input = self.calculate_fractal_input();
        let (signal, _) =
            self.generate_signal(self.previous_value, fractal_input, self.config.sensitivity);
        signal
    }
}

// Mathematical functions for Decimal
trait DecimalMath {
    fn sin(self) -> Option<Self>
    where
        Self: Sized;
    #[allow(dead_code)]
    fn cos(self) -> Option<Self>
    where
        Self: Sized;
}

impl DecimalMath for Decimal {
    fn sin(self) -> Option<Self> {
        // Taylor series approximation for sin(x)
        let x = self % (Decimal::new(2, 0) * Decimal::new(31416, 4)); // 2π approximation
        let mut result = x;
        let mut term = x;
        let x_squared = x * x;

        for n in 1..15 {
            let factorial_2n = Decimal::new((2 * n) as i64, 0);
            let factorial_2n_plus_1 = Decimal::new((2 * n + 1) as i64, 0);
            term *= -x_squared / (factorial_2n * factorial_2n_plus_1);
            result += term;

            if term.abs() < Decimal::new(1, 12) {
                // High precision threshold
                break;
            }
        }

        Some(result)
    }

    #[allow(dead_code)]
    fn cos(self) -> Option<Self> {
        // cos(x) = sin(x + π/2)
        let pi_half = Decimal::new(15708, 4); // π/2 ≈ 1.5708
        (self + pi_half).sin()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcr_chaos_creation() {
        let indicator = GCRChaosOscillator::default();
        assert!(!indicator.is_ready());
        assert_eq!(indicator.get_value(), Decimal::ZERO);
    }

    #[test]
    fn test_gcr_chaos_update() {
        let mut indicator = GCRChaosOscillator::default();

        // Add data points
        for i in 1..=20 {
            let price = Decimal::new(50000 + i * 100, 0);
            let volume = Decimal::new(1000 + i * 10, 0);
            let result = indicator.update(
                price + Decimal::new(50, 0),
                price - Decimal::new(50, 0),
                price,
                volume,
            );

            if i >= 14 {
                // Should be ready after lookback period
                assert!(result.is_some());
                let res = result.unwrap();
                assert!(res.value >= Decimal::new(-2, 0) && res.value <= Decimal::new(2, 0));
            }
        }

        assert!(indicator.is_ready());
    }

    #[test]
    fn test_signal_generation() {
        let mut indicator = GCRChaosOscillator::default();

        // Generate strong upward trend
        for i in 1..=20 {
            let price = Decimal::new(50000 + i * 500, 0); // Strong upward trend
            let volume = Decimal::new(2000, 0);
            let result = indicator.update(
                price + Decimal::new(200, 0),
                price - Decimal::new(200, 0),
                price,
                volume,
            );

            if let Some(res) = result {
                // Should eventually generate bullish signals in strong uptrend
                assert!(matches!(
                    res.signal,
                    GoldenSignal::Bull | GoldenSignal::HyperBull | GoldenSignal::Neutral
                ));
            }
        }
    }

    #[test]
    fn test_fractal_input_calculation() {
        let mut indicator = GCRChaosOscillator::default();

        // Add oscillating data
        for i in 1..=20 {
            let base_price = Decimal::new(50000, 0);
            let oscillation = if i % 2 == 0 {
                Decimal::new(500, 0)
            } else {
                Decimal::new(-500, 0)
            };
            let price = base_price + oscillation;
            let volume = Decimal::new(1000, 0);

            indicator.update(
                price + Decimal::new(100, 0),
                price - Decimal::new(100, 0),
                price,
                volume,
            );
        }

        assert!(indicator.is_ready());
        let fractal_input = indicator.calculate_fractal_input();
        // Should have some fractal input due to oscillation
        assert!(fractal_input.abs() > Decimal::ZERO);
    }

    #[test]
    fn test_reset_functionality() {
        let mut indicator = GCRChaosOscillator::default();

        // Add data
        for i in 1..=20 {
            let price = Decimal::new(50000 + i * 100, 0);
            let volume = Decimal::new(1000, 0);
            indicator.update(
                price + Decimal::new(50, 0),
                price - Decimal::new(50, 0),
                price,
                volume,
            );
        }

        assert!(indicator.is_ready());

        indicator.reset();
        assert!(!indicator.is_ready());
        assert_eq!(indicator.get_value(), Decimal::ZERO);
        assert_eq!(indicator.get_phase(), Decimal::ZERO);
        assert_eq!(indicator.price_history.len(), 0);
    }
}
