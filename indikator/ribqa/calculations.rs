//! RIBQA Calculations Module
//!
//! Matematiksel hesaplamalar ve algoritmalar

use crate::types::TechnicalAnalysisError;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
/// Mathematical utility functions for RIBQA calculations
mod math_utils {
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;

    /// Calculate square root using Newton's method
    pub fn sqrt(value: Decimal) -> Option<Decimal> {
        if value < Decimal::ZERO {
            return None;
        }
        if value == Decimal::ZERO {
            return Some(Decimal::ZERO);
        }

        let mut x = value / dec!(2.0);
        let mut prev_x = Decimal::ZERO;
        let precision = dec!(0.0001);

        for _ in 0..20 {
            // Max iterations
            if (x - prev_x).abs() < precision {
                break;
            }
            prev_x = x;
            x = (x + value / x) / dec!(2.0);
        }

        Some(x)
    }

    /// Calculate natural logarithm using Taylor series
    pub fn ln(value: Decimal) -> Option<Decimal> {
        if value <= Decimal::ZERO {
            return None;
        }
        if value == dec!(1.0) {
            return Some(Decimal::ZERO);
        }

        // Use change of base and approximation for small values
        let x = value - dec!(1.0);
        if x.abs() < dec!(0.5) {
            // Taylor series: ln(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...
            let mut result = x;
            let mut term = x;

            for n in 2..=10 {
                term = -term * x;
                result += term / Decimal::from(n);
            }

            Some(result)
        } else {
            // Simple approximation for larger values
            let f_val: f64 = value.try_into().unwrap_or(1.0);
            Some(Decimal::from_f64_retain(f_val.ln()).unwrap_or(Decimal::ZERO))
        }
    }

    /// Calculate hyperbolic tangent
    pub fn tanh(value: Decimal) -> Decimal {
        let f_val: f64 = (dec!(2.0) * value).try_into().unwrap_or(0.0);
        let exp_2x = f_val.exp();
        let result = (exp_2x - 1.0) / (exp_2x + 1.0);
        Decimal::from_f64_retain(result).unwrap_or(Decimal::ZERO)
    }
}

/// Turbulence calculation utilities
pub struct TurbulenceCalculator;

impl TurbulenceCalculator {
    /// Calculate enhanced turbulence with fractal dimension
    pub fn calculate_enhanced_turbulence(
        returns: &[Decimal],
    ) -> Result<Decimal, TechnicalAnalysisError> {
        if returns.is_empty() {
            return Ok(Decimal::ZERO);
        }

        let n = returns.len();

        // Calculate mean
        let mean = returns.iter().sum::<Decimal>() / Decimal::from(n);

        // Calculate variance and energy
        let mut variance = Decimal::ZERO;
        let mut energy = Decimal::ZERO;

        for &ret in returns {
            let deviation = ret - mean;
            variance += deviation * deviation;
            energy += ret * ret;
        }

        variance /= Decimal::from(n);
        let volatility = math_utils::sqrt(variance).unwrap_or(Decimal::ZERO);

        // Enhanced turbulence with complexity factor
        let complexity_factor = Self::calculate_complexity_factor(returns)?;
        let turbulence = volatility * energy * complexity_factor;

        Ok(turbulence.max(Decimal::ZERO))
    }

    /// Calculate complexity factor for turbulence enhancement
    pub fn calculate_complexity_factor(
        returns: &[Decimal],
    ) -> Result<Decimal, TechnicalAnalysisError> {
        if returns.len() < 3 {
            return Ok(dec!(1.0));
        }

        // Calculate local maxima and minima count
        let mut extrema_count = 0;

        for i in 1..returns.len() - 1 {
            let prev = returns[i - 1];
            let curr = returns[i];
            let next = returns[i + 1];

            // Local maximum or minimum
            if (curr > prev && curr > next) || (curr < prev && curr < next) {
                extrema_count += 1;
            }
        }

        // Normalize by window size
        let complexity = Decimal::from(extrema_count) / Decimal::from(returns.len());

        // Scale to reasonable range (1.0 to 2.0)
        Ok(dec!(1.0) + complexity)
    }
}

/// Recurrence calculation utilities
pub struct RecurrenceCalculator;

impl RecurrenceCalculator {
    /// Calculate enhanced recurrence with adaptive patterns
    pub fn calculate_enhanced_recurrence(
        returns: &[Decimal],
        threshold: Decimal,
    ) -> Result<Decimal, TechnicalAnalysisError> {
        if returns.len() < 3 {
            return Ok(Decimal::ZERO);
        }

        let mut recurrence_sum = Decimal::ZERO;
        let mut pattern_count = 0;

        // Multi-scale pattern detection
        for scale in 1..=3 {
            let scale_recurrence = Self::calculate_scale_recurrence(returns, threshold, scale)?;
            recurrence_sum += scale_recurrence / Decimal::from(scale); // Weight by scale
            pattern_count += 1;
        }

        if pattern_count == 0 {
            return Ok(Decimal::ZERO);
        }

        let avg_recurrence = recurrence_sum / Decimal::from(pattern_count);

        // Apply hyperbolic tangent normalization
        Ok(math_utils::tanh(avg_recurrence))
    }

    /// Calculate recurrence at specific scale
    pub fn calculate_scale_recurrence(
        returns: &[Decimal],
        threshold: Decimal,
        scale: usize,
    ) -> Result<Decimal, TechnicalAnalysisError> {
        if returns.len() < scale + 1 {
            return Ok(Decimal::ZERO);
        }

        let current_idx = returns.len() - 1;
        let current_pattern = &returns[current_idx - scale + 1..=current_idx];

        let mut similarity_sum = Decimal::ZERO;
        let mut comparisons = 0;

        // Compare with historical patterns
        for i in scale..current_idx {
            if i + scale <= returns.len() {
                let historical_pattern = &returns[i - scale + 1..=i];
                let similarity = Self::calculate_pattern_similarity(
                    current_pattern,
                    historical_pattern,
                    threshold,
                )?;

                // Weight recent patterns more heavily
                let time_weight = dec!(1.0) / Decimal::from(current_idx - i + 1);
                similarity_sum += similarity * time_weight;
                comparisons += 1;
            }
        }

        if comparisons == 0 {
            return Ok(Decimal::ZERO);
        }

        Ok(similarity_sum / Decimal::from(comparisons))
    }

    /// Calculate pattern similarity
    pub fn calculate_pattern_similarity(
        pattern1: &[Decimal],
        pattern2: &[Decimal],
        threshold: Decimal,
    ) -> Result<Decimal, TechnicalAnalysisError> {
        if pattern1.len() != pattern2.len() {
            return Ok(Decimal::ZERO);
        }

        let mut similarity = Decimal::ZERO;

        for (a, b) in pattern1.iter().zip(pattern2.iter()) {
            let diff = (a - b).abs();
            if diff < threshold {
                similarity += dec!(1.0) - (diff / threshold);
            }
        }

        Ok(similarity / Decimal::from(pattern1.len()))
    }
}

/// Fractal dimension calculation utilities
pub struct FractalCalculator;

impl FractalCalculator {
    /// Calculate fractal dimension using box-counting method
    pub fn calculate_fractal_dimension(
        returns: &[Decimal],
    ) -> Result<Decimal, TechnicalAnalysisError> {
        if returns.len() < 4 {
            return Ok(dec!(1.5)); // Default fractal dimension
        }

        // Simplified fractal dimension calculation
        let mut cumulative = vec![Decimal::ZERO; returns.len() + 1];
        for (i, &ret) in returns.iter().enumerate() {
            cumulative[i + 1] = cumulative[i] + ret;
        }

        // Calculate range and average deviation
        let max_cum = cumulative.iter().max().unwrap_or(&Decimal::ZERO);
        let min_cum = cumulative.iter().min().unwrap_or(&Decimal::ZERO);
        let range = max_cum - min_cum;

        if range == Decimal::ZERO {
            return Ok(dec!(1.0));
        }

        // Simplified fractal calculation
        let n = Decimal::from(returns.len());
        let log_n = math_utils::ln(n).unwrap_or(dec!(1.0));
        let log_range = math_utils::ln(range).unwrap_or(dec!(1.0));

        let fractal_dim = dec!(2.0) - (log_range / log_n);

        // Clamp between 1.0 and 2.0
        Ok(fractal_dim.max(dec!(1.0)).min(dec!(2.0)))
    }

    /// Calculate Hurst exponent for trend persistence
    pub fn calculate_hurst_exponent(
        returns: &[Decimal],
    ) -> Result<Decimal, TechnicalAnalysisError> {
        if returns.len() < 8 {
            return Ok(dec!(0.5)); // Random walk default
        }

        // R/S analysis for Hurst exponent
        let n = returns.len();
        let mean = returns.iter().sum::<Decimal>() / Decimal::from(n);

        // Calculate deviations from mean
        let deviations: Vec<Decimal> = returns.iter().map(|&r| r - mean).collect();

        // Calculate cumulative deviations
        let mut cumulative = vec![Decimal::ZERO; n + 1];
        for (i, &dev) in deviations.iter().enumerate() {
            cumulative[i + 1] = cumulative[i] + dev;
        }

        // Calculate range
        let max_cum = cumulative.iter().max().unwrap_or(&Decimal::ZERO);
        let min_cum = cumulative.iter().min().unwrap_or(&Decimal::ZERO);
        let range = max_cum - min_cum;

        // Calculate standard deviation
        let variance = deviations.iter().map(|&d| d * d).sum::<Decimal>() / Decimal::from(n);
        let std_dev = math_utils::sqrt(variance).unwrap_or(dec!(1.0));

        if std_dev == Decimal::ZERO {
            return Ok(dec!(0.5));
        }

        let rs_ratio = range / std_dev;
        let log_n = math_utils::ln(Decimal::from(n)).unwrap_or(dec!(1.0));
        let log_rs = math_utils::ln(rs_ratio).unwrap_or(dec!(1.0));

        let hurst = log_rs / log_n;

        // Clamp between 0.0 and 1.0
        Ok(hurst.max(Decimal::ZERO).min(dec!(1.0)))
    }
}

/// Volume analysis utilities
pub struct VolumeCalculator;

impl VolumeCalculator {
    /// Calculate volume confirmation factor
    pub fn calculate_volume_factor(volumes: &[Decimal]) -> Result<Decimal, TechnicalAnalysisError> {
        if volumes.len() < 2 {
            return Ok(dec!(1.0));
        }

        // Calculate volume trend
        let recent_vol = volumes.iter().rev().take(3).sum::<Decimal>() / dec!(3.0);
        let older_vol = volumes.iter().take(volumes.len() - 3).sum::<Decimal>()
            / Decimal::from(volumes.len() - 3);

        if older_vol == Decimal::ZERO {
            return Ok(dec!(1.0));
        }

        let vol_ratio = recent_vol / older_vol;

        // Volume confirmation: higher recent volume increases factor
        let factor = if vol_ratio > dec!(1.2) {
            dec!(1.2) // Strong volume confirmation
        } else if vol_ratio > dec!(1.0) {
            dec!(1.0) + (vol_ratio - dec!(1.0)) * dec!(2.0) // Moderate confirmation
        } else {
            dec!(0.8) + vol_ratio * dec!(0.2) // Weak volume
        };

        Ok(factor.max(dec!(0.5)).min(dec!(1.5)))
    }

    /// Calculate volume-weighted average price movement
    pub fn calculate_vwap_momentum(
        prices: &[Decimal],
        volumes: &[Decimal],
    ) -> Result<Decimal, TechnicalAnalysisError> {
        if prices.len() != volumes.len() || prices.is_empty() {
            return Ok(Decimal::ZERO);
        }

        let mut total_volume = Decimal::ZERO;
        let mut weighted_price_sum = Decimal::ZERO;

        for (price, volume) in prices.iter().zip(volumes.iter()) {
            total_volume += volume;
            weighted_price_sum += price * volume;
        }

        if total_volume == Decimal::ZERO {
            return Ok(Decimal::ZERO);
        }

        let vwap = weighted_price_sum / total_volume;
        let current_price = *prices.last().unwrap();

        // Return momentum relative to VWAP
        Ok((current_price - vwap) / vwap)
    }
}
