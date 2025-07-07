//! RIBQA Configuration Module
//!
//! RIBQA algoritması için konfigürasyon parametreleri ve ayarları

use rust_decimal::Decimal;
use rust_decimal_macros::dec;

/// RIBQA Configuration
#[derive(Debug, Clone)]
pub struct RibqaConfig {
    /// Analysis window (typically 14-21)
    pub window: usize,
    /// Recurrence threshold for pattern detection (typically 0.005)
    pub threshold: Decimal,
    /// Turbulence threshold for trend detection (typically 0.025)
    pub turbulence_threshold: Decimal,
    /// Recurrence threshold for range detection (typically 0.6)
    pub recurrence_threshold: Decimal,
    /// Volume confirmation enabled
    pub volume_confirmation: bool,
    /// Adaptive thresholds enabled
    pub adaptive_thresholds: bool,
    /// Multi-timeframe analysis enabled
    pub multi_timeframe: bool,
}

impl Default for RibqaConfig {
    fn default() -> Self {
        Self {
            window: 14,
            threshold: dec!(0.005),
            turbulence_threshold: dec!(0.025),
            recurrence_threshold: dec!(0.6),
            volume_confirmation: true,
            adaptive_thresholds: true,
            multi_timeframe: false, // Expensive, default off
        }
    }
}

impl RibqaConfig {
    /// Create new RIBQA configuration with custom parameters
    pub fn new(
        window: usize,
        threshold: Decimal,
        turbulence_threshold: Decimal,
        recurrence_threshold: Decimal,
    ) -> Self {
        Self {
            window,
            threshold,
            turbulence_threshold,
            recurrence_threshold,
            volume_confirmation: true,
            adaptive_thresholds: true,
            multi_timeframe: false,
        }
    }

    /// Create conservative configuration (less sensitive)
    pub fn conservative() -> Self {
        Self {
            window: 21,
            threshold: dec!(0.01),
            turbulence_threshold: dec!(0.035),
            recurrence_threshold: dec!(0.7),
            volume_confirmation: true,
            adaptive_thresholds: false,
            multi_timeframe: false,
        }
    }

    /// Create aggressive configuration (more sensitive)
    pub fn aggressive() -> Self {
        Self {
            window: 10,
            threshold: dec!(0.003),
            turbulence_threshold: dec!(0.015),
            recurrence_threshold: dec!(0.5),
            volume_confirmation: true,
            adaptive_thresholds: true,
            multi_timeframe: true,
        }
    }

    /// Enable volume confirmation
    pub fn with_volume_confirmation(mut self, enabled: bool) -> Self {
        self.volume_confirmation = enabled;
        self
    }

    /// Enable adaptive thresholds
    pub fn with_adaptive_thresholds(mut self, enabled: bool) -> Self {
        self.adaptive_thresholds = enabled;
        self
    }

    /// Enable multi-timeframe analysis
    pub fn with_multi_timeframe(mut self, enabled: bool) -> Self {
        self.multi_timeframe = enabled;
        self
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.window < 5 {
            return Err(format!("Window must be at least 5, got: {}", self.window));
        }

        if self.threshold <= Decimal::ZERO {
            return Err(format!(
                "Threshold must be positive, got: {}",
                self.threshold
            ));
        }

        if self.turbulence_threshold <= Decimal::ZERO {
            return Err(format!(
                "Turbulence threshold must be positive, got: {}",
                self.turbulence_threshold
            ));
        }

        if self.recurrence_threshold <= Decimal::ZERO || self.recurrence_threshold > dec!(1.0) {
            return Err(format!(
                "Recurrence threshold must be between 0 and 1, got: {}",
                self.recurrence_threshold
            ));
        }

        Ok(())
    }
}
