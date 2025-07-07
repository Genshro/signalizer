//! RIBQA Signals Module
//!
//! Sinyal üretimi ve trading sinyalleri

use super::analyzer::RibqaAnalyzer;
use super::types::{MarketRegime, RibqaResult};
use crate::types::{Signal, SignalData, SignalStrength, TechnicalAnalysisError};
use chrono::{DateTime, Utc};
use rust_decimal::prelude::*;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// RIBQA Signal Generator
pub struct RibqaSignalGenerator;

impl RibqaSignalGenerator {
    /// Generate trading signals based on RIBQA analysis
    pub fn generate_signal(
        analyzer: &RibqaAnalyzer,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        if let Some(current) = analyzer.current_result() {
            // Regime-based signal generation
            match current.market_regime {
                MarketRegime::Trending => Self::generate_trend_signal(analyzer, current, timestamp),
                MarketRegime::RangeBound => {
                    Self::generate_range_signal(analyzer, current, timestamp)
                },
                MarketRegime::Chaotic => {
                    // Avoid trading in chaotic conditions
                    Ok(None)
                },
                MarketRegime::Consolidation => {
                    Self::generate_breakout_signal(analyzer, current, timestamp)
                },
                MarketRegime::Transition => Self::generate_transition_signal(current, timestamp),
            }
        } else {
            Ok(None)
        }
    }

    /// Generate trend-following signals
    fn generate_trend_signal(
        analyzer: &RibqaAnalyzer,
        result: &RibqaResult,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        let config = analyzer.config();

        if result.turbulence > config.turbulence_threshold
            && result.hurst_exponent > dec!(0.6)
            && result.volume_factor > dec!(1.0)
        {
            // Determine trend direction from RIBQA value
            let ribqa_direction = result.ribqa_value.signum();

            if ribqa_direction != Decimal::ZERO {
                let signal_type = if ribqa_direction > Decimal::ZERO {
                    Signal::Buy
                } else {
                    Signal::Sell
                };

                let strength = result.get_signal_strength();
                let confidence = (result.turbulence * result.volume_factor * result.hurst_exponent)
                    .min(dec!(0.95));

                return Ok(Some(SignalData::new(
                    signal_type,
                    strength,
                    confidence,
                    timestamp,
                    "RIBQA_Trend".to_string(),
                )?));
            }
        }

        Ok(None)
    }

    /// Generate range-bound signals
    fn generate_range_signal(
        analyzer: &RibqaAnalyzer,
        result: &RibqaResult,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        let config = analyzer.config();

        if result.recurrence > config.recurrence_threshold
            && result.fractal_dimension < dec!(1.3)
            && result.volume_factor > dec!(0.9)
        {
            // Range trading: use RIBQA value to determine position within range
            let ribqa_magnitude = result.ribqa_value.abs();

            if ribqa_magnitude > dec!(0.02) {
                // Strong enough signal for range trading
                let (signal_type, strength) = if result.ribqa_value < dec!(-0.03) {
                    // Oversold in range - buy signal
                    (Signal::Buy, SignalStrength::Moderate)
                } else if result.ribqa_value > dec!(0.03) {
                    // Overbought in range - sell signal
                    (Signal::Sell, SignalStrength::Moderate)
                } else {
                    return Ok(None); // Middle of range
                };

                let confidence = result.recurrence * result.volume_factor;

                return Ok(Some(SignalData::new(
                    signal_type,
                    strength,
                    confidence,
                    timestamp,
                    "RIBQA_Range".to_string(),
                )?));
            }
        }

        Ok(None)
    }

    /// Generate breakout signals from consolidation
    fn generate_breakout_signal(
        analyzer: &RibqaAnalyzer,
        result: &RibqaResult,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        // Look for sudden increase in turbulence after consolidation
        if result.turbulence > dec!(0.02)
            && result.volume_factor > dec!(1.3)
            && analyzer.regime_persistence() > 5
        {
            // Was consolidating for a while

            let ribqa_magnitude = result.ribqa_value.abs();

            if ribqa_magnitude > dec!(0.01) {
                // Significant movement
                let signal_type = if result.ribqa_value > Decimal::ZERO {
                    Signal::Buy
                } else {
                    Signal::Sell
                };

                let confidence =
                    (result.turbulence * result.volume_factor * ribqa_magnitude).min(dec!(0.9));

                return Ok(Some(SignalData::new(
                    signal_type,
                    SignalStrength::Strong,
                    confidence,
                    timestamp,
                    "RIBQA_Breakout".to_string(),
                )?));
            }
        }

        Ok(None)
    }

    /// Generate conservative transition signals
    fn generate_transition_signal(
        result: &RibqaResult,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        // Conservative signals only during transition
        if result.ribqa_value.abs() > dec!(0.05) && result.volume_factor > dec!(1.1) {
            let signal_type = if result.ribqa_value > Decimal::ZERO {
                Signal::Buy
            } else {
                Signal::Sell
            };

            let confidence = (result.ribqa_value.abs() * result.volume_factor).min(dec!(0.8));

            return Ok(Some(SignalData::new(
                signal_type,
                SignalStrength::Weak,
                confidence,
                timestamp,
                "RIBQA_Transition".to_string(),
            )?));
        }

        Ok(None)
    }

    /// Generate multi-timeframe signals (advanced)
    pub fn generate_multi_timeframe_signal(
        short_term: &RibqaResult,
        medium_term: &RibqaResult,
        long_term: &RibqaResult,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        // Alignment check across timeframes
        let short_direction = short_term.ribqa_value.signum();
        let medium_direction = medium_term.ribqa_value.signum();
        let long_direction = long_term.ribqa_value.signum();

        // All timeframes must agree
        if short_direction == medium_direction
            && medium_direction == long_direction
            && short_direction != Decimal::ZERO
        {
            // Check regime alignment
            let regimes_aligned = matches!(
                (
                    short_term.market_regime,
                    medium_term.market_regime,
                    long_term.market_regime
                ),
                (
                    MarketRegime::Trending,
                    MarketRegime::Trending,
                    MarketRegime::Trending
                ) | (
                    MarketRegime::RangeBound,
                    MarketRegime::RangeBound,
                    MarketRegime::RangeBound
                )
            );

            if regimes_aligned {
                let signal_type = if short_direction > Decimal::ZERO {
                    Signal::Buy
                } else {
                    Signal::Sell
                };

                // Combine confidence from all timeframes
                let combined_confidence = (short_term.get_confidence() * dec!(0.5)
                    + medium_term.get_confidence() * dec!(0.3)
                    + long_term.get_confidence() * dec!(0.2))
                .min(dec!(0.95));

                // Stronger signal due to multi-timeframe alignment
                let strength = if combined_confidence > dec!(0.8) {
                    SignalStrength::VeryStrong
                } else if combined_confidence > dec!(0.6) {
                    SignalStrength::Strong
                } else {
                    SignalStrength::Moderate
                };

                return Ok(Some(SignalData::new(
                    signal_type,
                    strength,
                    combined_confidence,
                    timestamp,
                    "RIBQA_MultiTimeframe".to_string(),
                )?));
            }
        }

        Ok(None)
    }

    /// Generate divergence signals
    pub fn generate_divergence_signal(
        price_trend: Decimal, // 1 for up, -1 for down, 0 for sideways
        ribqa_current: &RibqaResult,
        ribqa_previous: &RibqaResult,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        let ribqa_trend = (ribqa_current.ribqa_value - ribqa_previous.ribqa_value).signum();

        // Detect divergence
        let is_bullish_divergence = price_trend < Decimal::ZERO && ribqa_trend > Decimal::ZERO;
        let is_bearish_divergence = price_trend > Decimal::ZERO && ribqa_trend < Decimal::ZERO;

        if is_bullish_divergence || is_bearish_divergence {
            let signal_type = if is_bullish_divergence {
                Signal::Buy
            } else {
                Signal::Sell
            };

            // Divergence strength based on magnitude
            let divergence_strength = (price_trend.abs() + ribqa_trend.abs()) / dec!(2.0);
            let confidence = (divergence_strength * ribqa_current.volume_factor).min(dec!(0.85));

            let strength = if confidence > dec!(0.7) {
                SignalStrength::Strong
            } else if confidence > dec!(0.5) {
                SignalStrength::Moderate
            } else {
                SignalStrength::Weak
            };

            return Ok(Some(SignalData::new(
                signal_type,
                strength,
                confidence,
                timestamp,
                "RIBQA_Divergence".to_string(),
            )?));
        }

        Ok(None)
    }

    /// Generate regime change signals
    pub fn generate_regime_change_signal(
        current_regime: MarketRegime,
        previous_regime: MarketRegime,
        result: &RibqaResult,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<SignalData>, TechnicalAnalysisError> {
        // Detect significant regime changes
        let significant_change = match (previous_regime, current_regime) {
            (MarketRegime::Consolidation, MarketRegime::Trending) => true,
            (MarketRegime::RangeBound, MarketRegime::Trending) => true,
            (MarketRegime::Chaotic, MarketRegime::Trending) => true,
            (MarketRegime::Trending, MarketRegime::RangeBound) => true,
            _ => false,
        };

        if significant_change && result.volume_factor > dec!(1.2) {
            let signal_type = match current_regime {
                MarketRegime::Trending => {
                    if result.ribqa_value > Decimal::ZERO {
                        Signal::Buy
                    } else {
                        Signal::Sell
                    }
                },
                MarketRegime::RangeBound => {
                    // Mean reversion signal
                    if result.ribqa_value > dec!(0.02) {
                        Signal::Sell
                    } else if result.ribqa_value < dec!(-0.02) {
                        Signal::Buy
                    } else {
                        return Ok(None);
                    }
                },
                _ => return Ok(None),
            };

            let confidence = result.get_confidence() * dec!(0.8); // Slightly reduced for regime change

            return Ok(Some(SignalData::new(
                signal_type,
                SignalStrength::Moderate,
                confidence,
                timestamp,
                "RIBQA_RegimeChange".to_string(),
            )?));
        }

        Ok(None)
    }
}
