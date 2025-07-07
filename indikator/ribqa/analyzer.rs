//! RIBQA Analyzer Module
//!
//! Ana RIBQA analiz motoru - veri işleme ve sonuç üretimi

use super::calculations::{
    FractalCalculator, RecurrenceCalculator, TurbulenceCalculator, VolumeCalculator,
};
use super::config::RibqaConfig;
use super::types::{MarketRegime, RibqaMetrics, RibqaResult};
use crate::types::{IndicatorResult, OhlcData, TechnicalAnalysisError};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::VecDeque;

/// RIBQA Indicator Implementation
#[derive(Debug, Clone)]
pub struct RibqaAnalyzer {
    config: RibqaConfig,
    ohlc_history: VecDeque<OhlcData>,
    returns: VecDeque<Decimal>,
    volume_history: VecDeque<Decimal>,
    turbulence_history: VecDeque<Decimal>,
    recurrence_history: VecDeque<Decimal>,
    current_result: Option<RibqaResult>,
    previous_result: Option<RibqaResult>,
    adaptive_threshold: Decimal,
    regime_persistence: usize,
    metrics: RibqaMetrics,
}

impl RibqaAnalyzer {
    /// Create new RIBQA analyzer
    pub fn new(config: RibqaConfig) -> Result<Self, TechnicalAnalysisError> {
        config
            .validate()
            .map_err(|e| TechnicalAnalysisError::invalid_config(e))?;

        Ok(Self {
            adaptive_threshold: config.threshold,
            config,
            ohlc_history: VecDeque::with_capacity(200),
            returns: VecDeque::with_capacity(200),
            volume_history: VecDeque::with_capacity(200),
            turbulence_history: VecDeque::with_capacity(50),
            recurrence_history: VecDeque::with_capacity(50),
            current_result: None,
            previous_result: None,
            regime_persistence: 0,
            metrics: RibqaMetrics::new(),
        })
    }

    /// Create RIBQA with default configuration
    pub fn default() -> Result<Self, TechnicalAnalysisError> {
        Self::new(RibqaConfig::default())
    }

    /// Update RIBQA with new OHLC data
    pub fn update(
        &mut self,
        ohlc: &OhlcData,
    ) -> Result<Option<IndicatorResult>, TechnicalAnalysisError> {
        // Add OHLC to history
        self.ohlc_history.push_back(ohlc.clone());
        self.volume_history.push_back(ohlc.volume);

        // Calculate logarithmic return if we have previous price
        if self.ohlc_history.len() > 1 {
            let prev_close = self.ohlc_history[self.ohlc_history.len() - 2].close;
            let current_close = ohlc.close;

            if prev_close > Decimal::ZERO {
                // Simple log approximation for returns
                let ratio = current_close / prev_close;
                let log_return = if ratio > Decimal::ZERO {
                    let f_ratio: f64 = ratio.try_into().unwrap_or(1.0);
                    Decimal::from_f64_retain(f_ratio.ln()).unwrap_or(Decimal::ZERO)
                } else {
                    Decimal::ZERO
                };
                self.returns.push_back(log_return);
            }
        }

        // Keep limited history for performance
        if self.ohlc_history.len() > 200 {
            self.ohlc_history.pop_front();
        }
        if self.returns.len() > 200 {
            self.returns.pop_front();
        }
        if self.volume_history.len() > 200 {
            self.volume_history.pop_front();
        }

        // Calculate RIBQA if we have enough data
        if self.returns.len() >= self.config.window && self.ohlc_history.len() > self.config.window
        {
            let ribqa_result = self.calculate_ribqa(ohlc.timestamp)?;

            // Store previous result
            self.previous_result = self.current_result.clone();
            self.current_result = Some(ribqa_result.clone());

            // Update adaptive parameters
            if self.config.adaptive_thresholds {
                self.update_adaptive_parameters();
            }

            return Ok(Some(IndicatorResult::new(
                ribqa_result.ribqa_value,
                ohlc.timestamp,
                "RIBQA".to_string(),
            )));
        }

        Ok(None)
    }

    /// Calculate comprehensive RIBQA analysis
    fn calculate_ribqa(
        &mut self,
        timestamp: DateTime<Utc>,
    ) -> Result<RibqaResult, TechnicalAnalysisError> {
        let window = self.config.window;
        let returns_len = self.returns.len();

        if returns_len < window {
            return Err(TechnicalAnalysisError::insufficient_data(
                window,
                returns_len,
            ));
        }

        // Get window data
        let window_returns: Vec<Decimal> = self
            .returns
            .iter()
            .rev()
            .take(window)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let window_volumes: Vec<Decimal> = self
            .volume_history
            .iter()
            .rev()
            .take(window)
            .cloned()
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        // Calculate core components
        let turbulence = TurbulenceCalculator::calculate_enhanced_turbulence(&window_returns)?;
        let recurrence = RecurrenceCalculator::calculate_enhanced_recurrence(
            &window_returns,
            self.adaptive_threshold,
        )?;
        let fractal_dimension = FractalCalculator::calculate_fractal_dimension(&window_returns)?;
        let hurst_exponent = FractalCalculator::calculate_hurst_exponent(&window_returns)?;

        // Volume confirmation factor
        let volume_factor = if self.config.volume_confirmation {
            VolumeCalculator::calculate_volume_factor(&window_volumes)?
        } else {
            dec!(1.0)
        };

        // Market regime detection
        let market_regime =
            self.detect_market_regime(turbulence, recurrence, fractal_dimension, hurst_exponent);

        // Calculate main RIBQA value with regime factor
        let regime_factor = market_regime.get_factor();
        let ribqa_value = turbulence * recurrence * regime_factor * volume_factor;

        // Store in history for adaptive calculations
        self.turbulence_history.push_back(turbulence);
        self.recurrence_history.push_back(recurrence);

        if self.turbulence_history.len() > 50 {
            self.turbulence_history.pop_front();
        }
        if self.recurrence_history.len() > 50 {
            self.recurrence_history.pop_front();
        }

        // Update metrics
        self.update_metrics(turbulence, recurrence);

        Ok(RibqaResult::new(
            ribqa_value,
            turbulence,
            recurrence,
            fractal_dimension,
            hurst_exponent,
            market_regime,
            volume_factor,
            timestamp,
        ))
    }

    /// Detect market regime based on RIBQA components
    fn detect_market_regime(
        &mut self,
        turbulence: Decimal,
        recurrence: Decimal,
        fractal_dim: Decimal,
        hurst: Decimal,
    ) -> MarketRegime {
        let regime = if turbulence > self.config.turbulence_threshold
            && recurrence < dec!(0.4)
            && hurst > dec!(0.6)
        {
            MarketRegime::Trending
        } else if turbulence < dec!(0.015)
            && recurrence > self.config.recurrence_threshold
            && fractal_dim < dec!(1.3)
        {
            MarketRegime::RangeBound
        } else if turbulence > dec!(0.04) && recurrence > dec!(0.7) {
            MarketRegime::Chaotic
        } else if turbulence < dec!(0.01) && recurrence < dec!(0.3) {
            MarketRegime::Consolidation
        } else {
            MarketRegime::Transition
        };

        // Update regime persistence for stability
        if let Some(prev_result) = &self.current_result {
            if prev_result.market_regime == regime {
                self.regime_persistence += 1;
            } else {
                self.regime_persistence = 0;
            }
        }

        regime
    }

    /// Update adaptive parameters based on recent performance
    fn update_adaptive_parameters(&mut self) {
        if self.turbulence_history.len() < 10 {
            return;
        }

        // Adaptive threshold based on recent volatility
        let recent_turbulence: Decimal = self
            .turbulence_history
            .iter()
            .rev()
            .take(10)
            .sum::<Decimal>()
            / dec!(10.0);

        // Adjust threshold based on market conditions
        if recent_turbulence > dec!(0.05) {
            // High volatility: increase threshold
            self.adaptive_threshold = self.config.threshold * dec!(1.5);
        } else if recent_turbulence < dec!(0.01) {
            // Low volatility: decrease threshold
            self.adaptive_threshold = self.config.threshold * dec!(0.7);
        } else {
            // Normal volatility: gradually return to default
            self.adaptive_threshold = (self.adaptive_threshold + self.config.threshold) / dec!(2.0);
        }

        // Clamp adaptive threshold
        self.adaptive_threshold = self
            .adaptive_threshold
            .max(self.config.threshold * dec!(0.5))
            .min(self.config.threshold * dec!(2.0));
    }

    /// Update internal metrics
    fn update_metrics(&mut self, turbulence: Decimal, recurrence: Decimal) {
        self.metrics.turbulence_history.push(turbulence);
        self.metrics.recurrence_history.push(recurrence);
        self.metrics.adaptive_threshold = self.adaptive_threshold;
        self.metrics.regime_persistence = self.regime_persistence;

        // Keep limited history
        if self.metrics.turbulence_history.len() > 50 {
            self.metrics.turbulence_history.remove(0);
        }
        if self.metrics.recurrence_history.len() > 50 {
            self.metrics.recurrence_history.remove(0);
        }
    }

    /// Get current RIBQA result
    pub fn current_result(&self) -> Option<&RibqaResult> {
        self.current_result.as_ref()
    }

    /// Get previous RIBQA result
    pub fn previous_result(&self) -> Option<&RibqaResult> {
        self.previous_result.as_ref()
    }

    /// Get current market regime
    pub fn current_regime(&self) -> Option<MarketRegime> {
        self.current_result.as_ref().map(|r| r.market_regime)
    }

    /// Check if RIBQA indicates strong trend
    pub fn is_strong_trend(&self) -> bool {
        if let Some(result) = &self.current_result {
            result.is_trending()
                && result.turbulence > self.config.turbulence_threshold * dec!(1.5)
                && result.hurst_exponent > dec!(0.7)
        } else {
            false
        }
    }

    /// Check if RIBQA indicates range market
    pub fn is_range_market(&self) -> bool {
        if let Some(result) = &self.current_result {
            result.is_range_bound()
                && result.recurrence > self.config.recurrence_threshold
                && result.fractal_dimension < dec!(1.4)
        } else {
            false
        }
    }

    /// Get RIBQA configuration
    pub fn config(&self) -> &RibqaConfig {
        &self.config
    }

    /// Get current metrics
    pub fn metrics(&self) -> &RibqaMetrics {
        &self.metrics
    }

    /// Reset the analyzer
    pub fn reset(&mut self) {
        self.ohlc_history.clear();
        self.returns.clear();
        self.volume_history.clear();
        self.turbulence_history.clear();
        self.recurrence_history.clear();
        self.current_result = None;
        self.previous_result = None;
        self.adaptive_threshold = self.config.threshold;
        self.regime_persistence = 0;
        self.metrics = RibqaMetrics::new();
    }

    /// Check if analyzer is ready
    pub fn is_ready(&self) -> bool {
        self.current_result.is_some()
    }

    /// Get minimum data points required
    pub fn min_data_points(&self) -> usize {
        self.config.window + 5 // Extra buffer for calculations
    }

    /// Get data history length
    pub fn history_len(&self) -> usize {
        self.ohlc_history.len()
    }

    /// Get adaptive threshold
    pub fn adaptive_threshold(&self) -> Decimal {
        self.adaptive_threshold
    }

    /// Get regime persistence count
    pub fn regime_persistence(&self) -> usize {
        self.regime_persistence
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
        volume: Decimal,
        timestamp: DateTime<Utc>,
    ) -> OhlcData {
        OhlcData::new(open, high, low, close, volume, timestamp).unwrap()
    }

    #[test]
    fn test_ribqa_creation() {
        let config = RibqaConfig::default();
        let ribqa = RibqaAnalyzer::new(config).unwrap();
        assert_eq!(ribqa.config.window, 14);
        assert!(!ribqa.is_ready());
    }

    #[test]
    fn test_ribqa_invalid_config() {
        let config = RibqaConfig {
            window: 2, // Too small
            ..Default::default()
        };
        assert!(RibqaAnalyzer::new(config).is_err());

        let config = RibqaConfig {
            threshold: dec!(0), // Invalid threshold
            ..Default::default()
        };
        assert!(RibqaAnalyzer::new(config).is_err());
    }

    #[test]
    fn test_ribqa_trending_market() {
        let mut ribqa = RibqaAnalyzer::default().unwrap();
        let base_time = Utc::now();

        // Create strong uptrend with increasing volume
        for i in 0..25 {
            let base_price = dec!(100) + Decimal::from(i * 2); // Strong trend
            let volume = dec!(1000) + Decimal::from(i * 50); // Increasing volume

            let ohlc = create_test_ohlc(
                base_price,
                base_price + dec!(3),
                base_price - dec!(1),
                base_price + dec!(2),
                volume,
                base_time + chrono::Duration::seconds(i),
            );
            let _ = ribqa.update(&ohlc);
        }

        assert!(ribqa.is_ready());

        if let Some(result) = ribqa.current_result() {
            // Should detect some market activity
            assert!(result.turbulence >= dec!(0.0));
            assert!(result.volume_factor > dec!(0.0));
        }
    }

    #[test]
    fn test_ribqa_range_market() {
        let mut ribqa = RibqaAnalyzer::default().unwrap();
        let base_time = Utc::now();

        // Create range-bound market
        let base_price = dec!(100);
        for i in 0..25 {
            let price_offset = (i % 6) as f64 - 2.5; // Oscillate between -2.5 and +2.5
            let close = base_price + Decimal::from_f64_retain(price_offset).unwrap();

            let ohlc = create_test_ohlc(
                close,
                close + dec!(1),
                close - dec!(1),
                close,
                dec!(1000),
                base_time + chrono::Duration::seconds(i),
            );
            let _ = ribqa.update(&ohlc);
        }

        assert!(ribqa.is_ready());

        if let Some(result) = ribqa.current_result() {
            // Should have some recurrence in range market
            assert!(result.recurrence >= dec!(0.0));
            assert!(result.fractal_dimension >= dec!(1.0));
        }
    }

    #[test]
    fn test_ribqa_reset() {
        let mut ribqa = RibqaAnalyzer::default().unwrap();
        let base_time = Utc::now();

        // Add some data
        for i in 0..10 {
            let ohlc = create_test_ohlc(
                dec!(100),
                dec!(101),
                dec!(99),
                dec!(100),
                dec!(1000),
                base_time + chrono::Duration::seconds(i),
            );
            let _ = ribqa.update(&ohlc);
        }

        assert!(ribqa.history_len() > 0);

        ribqa.reset();

        assert_eq!(ribqa.history_len(), 0);
        assert!(!ribqa.is_ready());
        assert!(ribqa.current_result().is_none());
    }

    #[test]
    fn test_ribqa_min_data_points() {
        let ribqa = RibqaAnalyzer::default().unwrap();
        assert_eq!(ribqa.min_data_points(), 19); // 14 + 5
    }

    #[test]
    fn test_ribqa_adaptive_thresholds() {
        let config = RibqaConfig {
            adaptive_thresholds: true,
            ..Default::default()
        };
        let mut ribqa = RibqaAnalyzer::new(config).unwrap();
        let base_time = Utc::now();

        let initial_threshold = ribqa.adaptive_threshold();

        // Generate high volatility data
        for i in 0..25 {
            let volatility = if i % 2 == 0 { dec!(10) } else { dec!(-10) };
            let close = dec!(100) + volatility;

            let ohlc = create_test_ohlc(
                close,
                close + dec!(1),
                close - dec!(1),
                close,
                dec!(1000),
                base_time + chrono::Duration::seconds(i),
            );
            let _ = ribqa.update(&ohlc);
        }

        // Adaptive threshold should have changed
        assert_ne!(ribqa.adaptive_threshold(), initial_threshold);
    }
}
