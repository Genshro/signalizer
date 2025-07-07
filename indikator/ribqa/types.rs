//! RIBQA Types Module
//!
//! RIBQA algoritması için veri tipleri, enum'lar ve result yapıları

use crate::types::SignalStrength;
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// RIBQA Market Regime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarketRegime {
    /// Strong trending market
    Trending,
    /// Range-bound market
    RangeBound,
    /// Transition between regimes
    Transition,
    /// Highly volatile/chaotic market
    Chaotic,
    /// Low activity/consolidation
    Consolidation,
}

impl MarketRegime {
    /// Get regime factor for RIBQA calculation
    pub fn get_factor(&self) -> Decimal {
        match self {
            MarketRegime::Trending => dec!(1.5), // Amplify trending signals
            MarketRegime::RangeBound => dec!(1.2), // Moderate range signals
            MarketRegime::Chaotic => dec!(0.7),  // Dampen chaotic signals
            MarketRegime::Consolidation => dec!(0.8), // Dampen low activity
            MarketRegime::Transition => dec!(1.0), // Neutral
        }
    }

    /// Get regime description
    pub fn description(&self) -> &'static str {
        match self {
            MarketRegime::Trending => "Strong directional movement",
            MarketRegime::RangeBound => "Price oscillating within range",
            MarketRegime::Transition => "Changing market conditions",
            MarketRegime::Chaotic => "High volatility and uncertainty",
            MarketRegime::Consolidation => "Low activity and tight range",
        }
    }
}

/// RIBQA Result containing all analysis components
#[derive(Debug, Clone)]
pub struct RibqaResult {
    /// Main RIBQA value (turbulence × recurrence × regime_factor)
    pub ribqa_value: Decimal,
    /// Market turbulence (volatility × energy × fractal_dimension)
    pub turbulence: Decimal,
    /// Pattern recurrence score (0-1)
    pub recurrence: Decimal,
    /// Fractal dimension (1-2, complexity measure)
    pub fractal_dimension: Decimal,
    /// Hurst exponent (0-1, trend persistence)
    pub hurst_exponent: Decimal,
    /// Detected market regime
    pub market_regime: MarketRegime,
    /// Volume confirmation factor (0-1)
    pub volume_factor: Decimal,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl RibqaResult {
    /// Create new RIBQA result
    pub fn new(
        ribqa_value: Decimal,
        turbulence: Decimal,
        recurrence: Decimal,
        fractal_dimension: Decimal,
        hurst_exponent: Decimal,
        market_regime: MarketRegime,
        volume_factor: Decimal,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            ribqa_value,
            turbulence,
            recurrence,
            fractal_dimension,
            hurst_exponent,
            market_regime,
            volume_factor,
            timestamp,
        }
    }

    /// Check if market is in trending regime
    pub fn is_trending(&self) -> bool {
        matches!(self.market_regime, MarketRegime::Trending)
    }

    /// Check if market is range-bound
    pub fn is_range_bound(&self) -> bool {
        matches!(self.market_regime, MarketRegime::RangeBound)
    }

    /// Check if market is chaotic
    pub fn is_chaotic(&self) -> bool {
        matches!(self.market_regime, MarketRegime::Chaotic)
    }

    /// Check if market is consolidating
    pub fn is_consolidating(&self) -> bool {
        matches!(self.market_regime, MarketRegime::Consolidation)
    }

    /// Check if market is in transition
    pub fn is_transitioning(&self) -> bool {
        matches!(self.market_regime, MarketRegime::Transition)
    }

    /// Get signal strength based on RIBQA components
    pub fn get_signal_strength(&self) -> SignalStrength {
        let strength_score = (self.ribqa_value * self.volume_factor).abs();

        if strength_score > dec!(0.1) {
            SignalStrength::VeryStrong
        } else if strength_score > dec!(0.05) {
            SignalStrength::Strong
        } else if strength_score > dec!(0.025) {
            SignalStrength::Moderate
        } else if strength_score > dec!(0.01) {
            SignalStrength::Weak
        } else {
            SignalStrength::VeryWeak
        }
    }

    /// Get confidence score (0-1)
    pub fn get_confidence(&self) -> Decimal {
        // Combine multiple factors for confidence
        let base_confidence = self.ribqa_value.abs();
        let volume_boost = if self.volume_factor > dec!(1.0) {
            (self.volume_factor - dec!(1.0)) * dec!(0.5)
        } else {
            Decimal::ZERO
        };

        let regime_boost = match self.market_regime {
            MarketRegime::Trending | MarketRegime::RangeBound => dec!(0.1),
            MarketRegime::Chaotic => dec!(-0.2),
            _ => Decimal::ZERO,
        };

        (base_confidence + volume_boost + regime_boost)
            .max(Decimal::ZERO)
            .min(dec!(1.0))
    }

    /// Check if this is a strong signal
    pub fn is_strong_signal(&self) -> bool {
        matches!(
            self.get_signal_strength(),
            SignalStrength::Strong | SignalStrength::VeryStrong
        )
    }

    /// Get regime factor
    pub fn get_regime_factor(&self) -> Decimal {
        self.market_regime.get_factor()
    }
}

/// RIBQA Analysis Metrics for detailed inspection
#[derive(Debug, Clone)]
pub struct RibqaMetrics {
    /// Volatility component
    pub volatility: Decimal,
    /// Energy component
    pub energy: Decimal,
    /// Complexity factor
    pub complexity_factor: Decimal,
    /// Pattern similarity scores
    pub pattern_similarities: Vec<Decimal>,
    /// Adaptive threshold used
    pub adaptive_threshold: Decimal,
    /// Regime persistence count
    pub regime_persistence: usize,
    /// Historical turbulence values
    pub turbulence_history: Vec<Decimal>,
    /// Historical recurrence values
    pub recurrence_history: Vec<Decimal>,
}

impl RibqaMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self {
            volatility: Decimal::ZERO,
            energy: Decimal::ZERO,
            complexity_factor: dec!(1.0),
            pattern_similarities: Vec::new(),
            adaptive_threshold: dec!(0.005),
            regime_persistence: 0,
            turbulence_history: Vec::new(),
            recurrence_history: Vec::new(),
        }
    }

    /// Get average pattern similarity
    pub fn avg_pattern_similarity(&self) -> Decimal {
        if self.pattern_similarities.is_empty() {
            return Decimal::ZERO;
        }

        self.pattern_similarities.iter().sum::<Decimal>()
            / Decimal::from(self.pattern_similarities.len())
    }

    /// Get turbulence trend (recent vs older)
    pub fn turbulence_trend(&self) -> Option<Decimal> {
        if self.turbulence_history.len() < 6 {
            return None;
        }

        let recent: Decimal = self
            .turbulence_history
            .iter()
            .rev()
            .take(3)
            .sum::<Decimal>()
            / dec!(3.0);

        let older: Decimal = self
            .turbulence_history
            .iter()
            .rev()
            .skip(3)
            .take(3)
            .sum::<Decimal>()
            / dec!(3.0);

        if older == Decimal::ZERO {
            return None;
        }

        Some((recent - older) / older)
    }
}
