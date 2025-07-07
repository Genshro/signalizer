//! Volume indicators module
//!
//! Contains indicators that analyze trading volume patterns and trends.
//! These indicators help identify the strength of price movements through volume analysis.

pub mod ad_line;
pub mod cmf;
pub mod klinger;
pub mod mfi;
pub mod obv;
pub mod pvt;
pub mod signals;
pub mod volume_profile;
pub mod vwap;

// Re-export main types for convenience
pub use ad_line::{
    AccumulationDistributionLine, AdLineConfig, AdLineResult, AdLineTrend, AdStatus,
    DivergenceType as AdDivergenceType,
};
pub use cmf::{ChaikinMoneyFlow, CmfConfig, CmfResult, MoneyFlowStrength};
pub use klinger::{
    KlingerConfig, KlingerOscillator, KlingerResult, KlingerTrend,
    MoneyFlowDirection as KlingerMoneyFlow,
};
pub use mfi::{MfiConfig, MfiResult, MoneyFlowDirection, MoneyFlowIndex};
pub use obv::{ObvConfig, ObvResult, OnBalanceVolume};
pub use pvt::{PriceMomentum, PriceVolumeTrend, PvtConfig, PvtResult, PvtTrend};
pub use signals::{
    DivergenceAnalysis, DivergenceType as SignalDivergenceType, IndicatorWeights, IndividualSignal,
    MarketPhase, VolumeConfirmation, VolumeSignalConfig, VolumeSignalGenerator, VolumeSignalResult,
    VolumeSignificance, VolumeTrend,
};
pub use volume_profile::{
    PricePosition, VolumeConcentration, VolumeProfile, VolumeProfileConfig, VolumeProfileResult,
};
pub use vwap::{VolumeWeightedAveragePrice, VwapBand, VwapConfig, VwapResult};

use crate::types::{IndicatorResult, OhlcData, SignalData, TechnicalAnalysisError};
use chrono::{DateTime, Utc};

/// Volume Indicators Collection
///
/// This struct provides a unified interface to all volume indicators,
/// allowing for easy batch processing and signal generation.
#[derive(Debug, Clone)]
pub struct VolumeIndicators {
    /// OBV (On-Balance Volume) indicator
    pub obv: Option<OnBalanceVolume>,
    /// MFI (Money Flow Index) indicator
    pub mfi: Option<MoneyFlowIndex>,
    /// VWAP (Volume-Weighted Average Price) indicator
    pub vwap: Option<VolumeWeightedAveragePrice>,
    /// CMF (Chaikin Money Flow) indicator
    pub cmf: Option<ChaikinMoneyFlow>,
    /// A/D Line (Accumulation/Distribution Line) indicator
    pub ad_line: Option<AccumulationDistributionLine>,
    /// PVT (Price Volume Trend) indicator
    pub pvt: Option<PriceVolumeTrend>,
    /// Volume Profile indicator
    pub volume_profile: Option<VolumeProfile>,
    /// Klinger Oscillator indicator
    pub klinger: Option<KlingerOscillator>,
}

impl VolumeIndicators {
    /// Create new volume indicators instance
    pub fn new() -> Self {
        Self {
            obv: None,
            mfi: None,
            vwap: None,
            cmf: None,
            ad_line: None,
            pvt: None,
            volume_profile: None,
            klinger: None,
        }
    }

    /// Add OBV indicator with default configuration
    pub fn with_obv(mut self) -> Result<Self, TechnicalAnalysisError> {
        self.obv = Some(OnBalanceVolume::default());
        Ok(self)
    }

    /// Add OBV indicator with custom configuration
    pub fn with_obv_config(mut self, config: ObvConfig) -> Result<Self, TechnicalAnalysisError> {
        self.obv = Some(OnBalanceVolume::new(config)?);
        Ok(self)
    }

    /// Add MFI indicator with default configuration
    pub fn with_mfi(mut self) -> Result<Self, TechnicalAnalysisError> {
        self.mfi = Some(MoneyFlowIndex::default());
        Ok(self)
    }

    /// Add MFI indicator with custom configuration
    pub fn with_mfi_config(mut self, config: MfiConfig) -> Result<Self, TechnicalAnalysisError> {
        self.mfi = Some(MoneyFlowIndex::new(config)?);
        Ok(self)
    }

    /// Add VWAP indicator with default configuration
    pub fn with_vwap(mut self) -> Result<Self, TechnicalAnalysisError> {
        self.vwap = Some(VolumeWeightedAveragePrice::default());
        Ok(self)
    }

    /// Add VWAP indicator with custom configuration
    pub fn with_vwap_config(mut self, config: VwapConfig) -> Result<Self, TechnicalAnalysisError> {
        self.vwap = Some(VolumeWeightedAveragePrice::new(config)?);
        Ok(self)
    }

    /// Add CMF indicator with default configuration
    pub fn with_cmf(mut self) -> Result<Self, TechnicalAnalysisError> {
        self.cmf = Some(ChaikinMoneyFlow::default());
        Ok(self)
    }

    /// Add CMF indicator with custom configuration
    pub fn with_cmf_config(mut self, config: CmfConfig) -> Result<Self, TechnicalAnalysisError> {
        self.cmf = Some(ChaikinMoneyFlow::new(config)?);
        Ok(self)
    }

    /// Add Klinger Oscillator with default configuration
    pub fn with_klinger(mut self) -> Result<Self, TechnicalAnalysisError> {
        self.klinger = Some(KlingerOscillator::default());
        Ok(self)
    }

    /// Add Klinger Oscillator with custom configuration
    pub fn with_klinger_config(
        mut self,
        config: KlingerConfig,
    ) -> Result<Self, TechnicalAnalysisError> {
        self.klinger = Some(KlingerOscillator::new(config)?);
        Ok(self)
    }

    /// Update all volume indicators with new OHLC data
    pub fn update(
        &mut self,
        ohlc: &OhlcData,
    ) -> Result<Vec<IndicatorResult>, TechnicalAnalysisError> {
        let mut results = Vec::new();

        // Update OBV
        if let Some(ref mut obv) = self.obv {
            if let Some(result) = obv.update(ohlc)? {
                results.push(result);
            }
        }

        // Update MFI
        if let Some(ref mut mfi) = self.mfi {
            if let Some(result) = mfi.update(ohlc)? {
                results.push(result);
            }
        }

        // Update VWAP
        if let Some(ref mut vwap) = self.vwap {
            if let Some(result) = vwap.update(ohlc)? {
                results.push(result);
            }
        }

        // Update CMF
        if let Some(ref mut cmf) = self.cmf {
            if let Some(result) = cmf.update(ohlc)? {
                results.push(result);
            }
        }

        // Update Klinger
        if let Some(ref mut klinger) = self.klinger {
            if let Some(result) = klinger.update(ohlc)? {
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Generate all volume signals
    pub fn generate_signals(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<Vec<SignalData>, TechnicalAnalysisError> {
        let mut signals = Vec::new();

        // Generate OBV signals
        if let Some(ref obv) = self.obv {
            if let Some(signal) = obv.generate_signal(timestamp)? {
                signals.push(signal);
            }
        }

        // Generate MFI signals
        if let Some(ref mfi) = self.mfi {
            if let Some(signal) = mfi.generate_signal(timestamp)? {
                signals.push(signal);
            }
        }

        // Generate VWAP signals
        if let Some(ref vwap) = self.vwap {
            if let Some(signal) = vwap.generate_signal(timestamp)? {
                signals.push(signal);
            }
        }

        // Generate CMF signals
        if let Some(ref cmf) = self.cmf {
            if let Some(signal) = cmf.generate_signal(timestamp)? {
                signals.push(signal);
            }
        }

        // Generate Klinger signals
        if let Some(ref klinger) = self.klinger {
            if let Some(signal) = klinger.generate_signal(timestamp)? {
                signals.push(signal);
            }
        }

        Ok(signals)
    }

    /// Get volume consensus (agreement between indicators)
    pub fn get_volume_consensus(&self) -> VolumeConsensus {
        let mut bullish_count = 0;
        let mut bearish_count = 0;
        let mut total_indicators = 0;

        // Check OBV
        if let Some(ref obv) = self.obv {
            total_indicators += 1;
            if obv.is_bullish() {
                bullish_count += 1;
            } else if obv.is_bearish() {
                bearish_count += 1;
            }
        }

        // Check MFI
        if let Some(ref mfi) = self.mfi {
            total_indicators += 1;
            if mfi.is_overbought() {
                bearish_count += 1;
            } else if mfi.is_oversold() {
                bullish_count += 1;
            }
        }

        // Check VWAP
        if let Some(ref vwap) = self.vwap {
            total_indicators += 1;
            if vwap.is_above_vwap() {
                bullish_count += 1;
            } else if vwap.is_below_vwap() {
                bearish_count += 1;
            }
        }

        // Check CMF
        if let Some(ref cmf) = self.cmf {
            total_indicators += 1;
            if cmf.is_positive_flow() {
                bullish_count += 1;
            } else if cmf.is_negative_flow() {
                bearish_count += 1;
            }
        }

        if total_indicators == 0 {
            return VolumeConsensus::Neutral;
        }

        // Calculate consensus based on agreement percentage
        let bullish_ratio = bullish_count as f64 / total_indicators as f64;
        let bearish_ratio = bearish_count as f64 / total_indicators as f64;

        if bullish_ratio >= 0.75 {
            VolumeConsensus::StrongBullish
        } else if bullish_ratio >= 0.5 {
            VolumeConsensus::WeakBullish
        } else if bearish_ratio >= 0.75 {
            VolumeConsensus::StrongBearish
        } else if bearish_ratio >= 0.5 {
            VolumeConsensus::WeakBearish
        } else {
            VolumeConsensus::Neutral
        }
    }

    /// Reset all volume indicators
    pub fn reset(&mut self) {
        if let Some(ref mut obv) = self.obv {
            obv.reset();
        }
        if let Some(ref mut mfi) = self.mfi {
            mfi.reset();
        }
        if let Some(ref mut vwap) = self.vwap {
            vwap.reset();
        }
        if let Some(ref mut cmf) = self.cmf {
            cmf.reset();
        }

        if let Some(ref mut klinger) = self.klinger {
            klinger.reset();
        }
    }

    /// Check if all enabled indicators are ready
    pub fn is_ready(&self) -> bool {
        let obv_ready = self.obv.as_ref().map_or(true, |obv| obv.is_ready());
        let mfi_ready = self.mfi.as_ref().map_or(true, |mfi| mfi.is_ready());
        let vwap_ready = self.vwap.as_ref().map_or(true, |vwap| vwap.is_ready());
        let cmf_ready = self.cmf.as_ref().map_or(true, |cmf| cmf.is_ready());

        obv_ready && mfi_ready && vwap_ready && cmf_ready
    }
}

impl Default for VolumeIndicators {
    fn default() -> Self {
        Self::new()
    }
}

/// Volume consensus types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeConsensus {
    /// Strong bullish volume consensus (75%+ agreement)
    StrongBullish,
    /// Weak bullish volume consensus (50-75% agreement)
    WeakBullish,
    /// Neutral volume consensus
    Neutral,
    /// Weak bearish volume consensus (50-75% agreement)
    WeakBearish,
    /// Strong bearish volume consensus (75%+ agreement)
    StrongBearish,
}

/// Volume strength levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeStrength {
    /// Very high volume (3x+ average)
    VeryHigh,
    /// High volume (2x+ average)
    High,
    /// Normal volume (0.5x-2x average)
    Normal,
    /// Low volume (below 0.5x average)
    Low,
}
