//! Money Flow Index (MFI) Indicator
//!
//! MFI is a momentum indicator that uses both price and volume to identify overbought
//! or oversold conditions. It's often called "Volume-weighted RSI" as it incorporates
//! volume into the RSI calculation.

use crate::types::{
    IndicatorResult, OhlcData, Price, Signal, SignalStrength, TechnicalAnalysisError, Timeframe,
};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::collections::{HashMap, VecDeque};

/// MFI Configuration
#[derive(Debug, Clone)]
pub struct MfiConfig {
    /// Period for MFI calculation (default: 14)
    pub period: usize,
    /// Overbought threshold (default: 80)
    pub overbought_threshold: Decimal,
    /// Oversold threshold (default: 20)
    pub oversold_threshold: Decimal,
    /// Enable divergence detection
    pub enable_divergence: bool,
}

impl Default for MfiConfig {
    fn default() -> Self {
        Self {
            period: 14,
            overbought_threshold: dec!(80),
            oversold_threshold: dec!(20),
            enable_divergence: true,
        }
    }
}

/// MFI calculation result
#[derive(Debug, Clone)]
pub struct MfiResult {
    /// Current MFI value (0-100)
    pub mfi_value: Decimal,
    /// Money flow direction
    pub direction: MoneyFlowDirection,
    /// Signal strength
    pub signal_strength: SignalStrength,
    /// Raw money flow
    pub raw_money_flow: Decimal,
    /// Positive money flow ratio
    pub positive_flow_ratio: Decimal,
    /// Divergence detection (if enabled)
    pub divergence: Option<MfiDivergence>,
}

/// Money flow direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoneyFlowDirection {
    /// Strong positive money flow
    StrongPositive,
    /// Moderate positive money flow
    ModeratePositive,
    /// Neutral money flow
    Neutral,
    /// Moderate negative money flow
    ModerateNegative,
    /// Strong negative money flow
    StrongNegative,
}

/// MFI divergence detection
#[derive(Debug, Clone)]
pub struct MfiDivergence {
    /// Divergence type
    pub divergence_type: DivergenceType,
    /// Divergence strength (0.0 to 1.0)
    pub strength: Decimal,
    /// Number of periods in divergence
    pub periods: usize,
}

/// Divergence types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivergenceType {
    /// Bullish divergence (price down, MFI up)
    Bullish,
    /// Bearish divergence (price up, MFI down)
    Bearish,
    /// No divergence
    None,
}

/// Money Flow Index indicator
#[derive(Debug, Clone)]
pub struct MoneyFlowIndex {
    /// Configuration
    config: MfiConfig,
    /// Typical price history
    typical_prices: VecDeque<Decimal>,
    /// Raw money flow history
    raw_money_flows: VecDeque<Decimal>,
    /// Positive money flows
    positive_flows: VecDeque<Decimal>,
    /// Negative money flows
    negative_flows: VecDeque<Decimal>,
    /// Price history for divergence detection
    price_history: VecDeque<Decimal>,
    /// MFI history
    mfi_history: VecDeque<Decimal>,
    /// Timestamps
    timestamps: VecDeque<DateTime<Utc>>,
    /// Is indicator ready
    is_ready: bool,
}

impl MoneyFlowIndex {
    /// Create new MFI indicator
    pub fn new(config: MfiConfig) -> Result<Self, TechnicalAnalysisError> {
        if config.period == 0 {
            return Err(TechnicalAnalysisError::invalid_config(
                "Period cannot be zero",
            ));
        }

        if config.overbought_threshold <= config.oversold_threshold {
            return Err(TechnicalAnalysisError::invalid_config(
                "Overbought threshold must be greater than oversold threshold",
            ));
        }

        Ok(Self {
            config,
            typical_prices: VecDeque::new(),
            raw_money_flows: VecDeque::new(),
            positive_flows: VecDeque::new(),
            negative_flows: VecDeque::new(),
            price_history: VecDeque::new(),
            mfi_history: VecDeque::new(),
            timestamps: VecDeque::new(),
            is_ready: false,
        })
    }

    /// Update MFI with new OHLC data
    pub fn update(
        &mut self,
        ohlc: &OhlcData,
    ) -> Result<Option<IndicatorResult>, TechnicalAnalysisError> {
        // Calculate typical price (HLC/3)
        let typical_price = (ohlc.high + ohlc.low + ohlc.close) / dec!(3);

        // Calculate raw money flow (typical price * volume)
        let raw_money_flow = typical_price * ohlc.volume;

        // Add to history
        self.typical_prices.push_back(typical_price);
        self.raw_money_flows.push_back(raw_money_flow);
        self.price_history.push_back(ohlc.close);
        self.timestamps.push_back(ohlc.timestamp);

        // Determine if money flow is positive or negative
        if self.typical_prices.len() > 1 {
            let current_tp = typical_price;
            let previous_tp = self.typical_prices[self.typical_prices.len() - 2];

            if current_tp > previous_tp {
                // Positive money flow
                self.positive_flows.push_back(raw_money_flow);
                self.negative_flows.push_back(dec!(0));
            } else if current_tp < previous_tp {
                // Negative money flow
                self.positive_flows.push_back(dec!(0));
                self.negative_flows.push_back(raw_money_flow);
            } else {
                // No change
                self.positive_flows.push_back(dec!(0));
                self.negative_flows.push_back(dec!(0));
            }
        } else {
            // First data point - neutral
            self.positive_flows.push_back(dec!(0));
            self.negative_flows.push_back(dec!(0));
        }

        // Keep limited history
        let max_history = self.config.period * 3;
        if self.typical_prices.len() > max_history {
            self.typical_prices.pop_front();
            self.raw_money_flows.pop_front();
            self.positive_flows.pop_front();
            self.negative_flows.pop_front();
            self.price_history.pop_front();
            self.timestamps.pop_front();
        }

        // Check if ready
        if self.typical_prices.len() >= self.config.period + 1 {
            self.is_ready = true;
        }

        if !self.is_ready {
            return Ok(None);
        }

        // Calculate MFI
        let mfi_value = self.calculate_mfi()?;
        self.mfi_history.push_back(mfi_value);

        // Keep MFI history limited
        if self.mfi_history.len() > max_history {
            self.mfi_history.pop_front();
        }

        // Generate result
        let result = self.generate_result(ohlc, mfi_value)?;
        Ok(Some(result))
    }

    /// Calculate MFI value
    fn calculate_mfi(&self) -> Result<Decimal, TechnicalAnalysisError> {
        if self.positive_flows.len() < self.config.period
            || self.negative_flows.len() < self.config.period
        {
            return Err(TechnicalAnalysisError::insufficient_data(
                self.config.period,
                self.positive_flows.len(),
            ));
        }

        // Sum positive and negative money flows over the period
        let positive_sum: Decimal = self
            .positive_flows
            .iter()
            .rev()
            .take(self.config.period)
            .sum();

        let negative_sum: Decimal = self
            .negative_flows
            .iter()
            .rev()
            .take(self.config.period)
            .sum();

        if negative_sum == dec!(0) {
            return Ok(dec!(100)); // All positive flow
        }

        // Calculate Money Flow Ratio
        let money_flow_ratio = positive_sum / negative_sum;

        // Calculate MFI
        let mfi = dec!(100) - (dec!(100) / (dec!(1) + money_flow_ratio));

        Ok(mfi)
    }

    /// Generate MFI result
    fn generate_result(
        &self,
        ohlc: &OhlcData,
        mfi_value: Decimal,
    ) -> Result<IndicatorResult, TechnicalAnalysisError> {
        let direction = self.calculate_direction(mfi_value);
        let signal_strength = self.calculate_signal_strength(mfi_value);
        let positive_flow_ratio = self.calculate_positive_flow_ratio();
        let raw_money_flow = self.raw_money_flows.back().cloned().unwrap_or(dec!(0));

        let divergence_data = if self.config.enable_divergence {
            self.detect_divergence()
        } else {
            None
        };

        let mfi_result = MfiResult {
            mfi_value,
            direction,
            signal_strength,
            raw_money_flow,
            positive_flow_ratio,
            divergence: divergence_data.clone(),
        };

        // Generate signal
        let signal = self.generate_signal_from_result(&mfi_result);

        // Create additional values
        let mut additional_values = HashMap::new();
        additional_values.insert("direction".to_string(), Decimal::from(direction as u8));
        additional_values.insert("raw_money_flow".to_string(), raw_money_flow);
        additional_values.insert("positive_flow_ratio".to_string(), positive_flow_ratio);
        additional_values.insert(
            "signal_strength".to_string(),
            Decimal::from(signal_strength as u8),
        );

        if let Some(ref div) = divergence_data {
            additional_values.insert(
                "divergence_type".to_string(),
                Decimal::from(div.divergence_type as u8),
            );
            additional_values.insert("divergence_strength".to_string(), div.strength);
        }

        let mut result = IndicatorResult::new(mfi_value, ohlc.timestamp, "MFI".to_string());

        result.additional_values = Some(additional_values);
        result.signal = Some(signal);

        Ok(result)
    }

    /// Calculate money flow direction
    fn calculate_direction(&self, mfi_value: Decimal) -> MoneyFlowDirection {
        if mfi_value >= dec!(80) {
            MoneyFlowDirection::StrongPositive
        } else if mfi_value >= dec!(60) {
            MoneyFlowDirection::ModeratePositive
        } else if mfi_value <= dec!(20) {
            MoneyFlowDirection::StrongNegative
        } else if mfi_value <= dec!(40) {
            MoneyFlowDirection::ModerateNegative
        } else {
            MoneyFlowDirection::Neutral
        }
    }

    /// Calculate signal strength
    fn calculate_signal_strength(&self, mfi_value: Decimal) -> SignalStrength {
        if mfi_value >= self.config.overbought_threshold
            || mfi_value <= self.config.oversold_threshold
        {
            SignalStrength::Strong
        } else if mfi_value >= dec!(70) || mfi_value <= dec!(30) {
            SignalStrength::Moderate
        } else {
            SignalStrength::Weak
        }
    }

    /// Calculate positive flow ratio
    fn calculate_positive_flow_ratio(&self) -> Decimal {
        if self.positive_flows.len() < self.config.period {
            return dec!(0.5);
        }

        let positive_sum: Decimal = self
            .positive_flows
            .iter()
            .rev()
            .take(self.config.period)
            .sum();

        let negative_sum: Decimal = self
            .negative_flows
            .iter()
            .rev()
            .take(self.config.period)
            .sum();

        let total = positive_sum + negative_sum;
        if total == dec!(0) {
            dec!(0.5)
        } else {
            positive_sum / total
        }
    }

    /// Detect divergence between price and MFI
    fn detect_divergence(&self) -> Option<MfiDivergence> {
        if self.mfi_history.len() < self.config.period * 2
            || self.price_history.len() < self.config.period * 2
        {
            return None;
        }

        let lookback = self.config.period;
        let recent_mfi: Vec<Decimal> = self
            .mfi_history
            .iter()
            .rev()
            .take(lookback)
            .cloned()
            .collect();

        let recent_prices: Vec<Decimal> = self
            .price_history
            .iter()
            .rev()
            .take(lookback)
            .cloned()
            .collect();

        // Calculate trends
        let mfi_trend = self.calculate_trend_direction(&recent_mfi);
        let price_trend = self.calculate_trend_direction(&recent_prices);

        let divergence_type = match (price_trend, mfi_trend) {
            (TrendDir::Down, TrendDir::Up) => DivergenceType::Bullish,
            (TrendDir::Up, TrendDir::Down) => DivergenceType::Bearish,
            _ => DivergenceType::None,
        };

        if divergence_type != DivergenceType::None {
            let strength = self.calculate_divergence_strength(&recent_prices, &recent_mfi);
            Some(MfiDivergence {
                divergence_type,
                strength,
                periods: lookback,
            })
        } else {
            None
        }
    }

    /// Calculate trend direction for divergence
    fn calculate_trend_direction(&self, values: &[Decimal]) -> TrendDir {
        if values.len() < 2 {
            return TrendDir::Sideways;
        }

        let first = values[values.len() - 1];
        let last = values[0];

        let change = (last - first) / first.abs();

        if change > dec!(0.05) {
            TrendDir::Up
        } else if change < dec!(-0.05) {
            TrendDir::Down
        } else {
            TrendDir::Sideways
        }
    }

    /// Calculate divergence strength
    fn calculate_divergence_strength(&self, prices: &[Decimal], mfi_values: &[Decimal]) -> Decimal {
        if prices.len() != mfi_values.len() || prices.len() < 2 {
            return dec!(0);
        }

        let price_change = (prices[0] - prices[prices.len() - 1]) / prices[prices.len() - 1].abs();
        let mfi_change = (mfi_values[0] - mfi_values[mfi_values.len() - 1])
            / mfi_values[mfi_values.len() - 1].abs();

        // Strength is based on how much the trends diverge
        (price_change - mfi_change).abs().min(dec!(1.0))
    }

    /// Generate signal from MFI result
    fn generate_signal_from_result(&self, result: &MfiResult) -> Signal {
        match result.direction {
            MoneyFlowDirection::StrongNegative
                if result.mfi_value <= self.config.oversold_threshold =>
            {
                Signal::StrongBuy
            },
            MoneyFlowDirection::ModerateNegative if result.mfi_value <= dec!(30) => Signal::Buy,
            MoneyFlowDirection::StrongPositive
                if result.mfi_value >= self.config.overbought_threshold =>
            {
                Signal::StrongSell
            },
            MoneyFlowDirection::ModeratePositive if result.mfi_value >= dec!(70) => Signal::Sell,
            _ => Signal::Neutral,
        }
    }

    /// Generate signal for external use
    pub fn generate_signal(
        &self,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<crate::types::SignalData>, TechnicalAnalysisError> {
        if !self.is_ready || self.mfi_history.is_empty() {
            return Ok(None);
        }

        let mfi_value = *self.mfi_history.back().unwrap();
        let direction = self.calculate_direction(mfi_value);
        let signal_strength = self.calculate_signal_strength(mfi_value);

        let signal = match direction {
            MoneyFlowDirection::StrongNegative if mfi_value <= self.config.oversold_threshold => {
                Signal::StrongBuy
            },
            MoneyFlowDirection::ModerateNegative if mfi_value <= dec!(30) => Signal::Buy,
            MoneyFlowDirection::StrongPositive if mfi_value >= self.config.overbought_threshold => {
                Signal::StrongSell
            },
            MoneyFlowDirection::ModeratePositive if mfi_value >= dec!(70) => Signal::Sell,
            _ => Signal::Neutral,
        };

        // Create metadata
        let metadata = serde_json::json!({
            "mfi_value": mfi_value.to_string(),
            "direction": format!("{:?}", direction)
        });

        let signal_data = crate::types::SignalData {
            signal,
            strength: signal_strength,
            confidence: dec!(0.7), // Default confidence
            timestamp,
            source: "MFI".to_string(),
            metadata: Some(metadata),
            price: self.price_history.back().copied(),
            volume: None, // MFI doesn't track volume separately
            timeframe: Some(Timeframe::H1),
            duration: Some(chrono::Duration::hours(1)),
            stop_loss: None,
            take_profit: None,
            risk_reward_ratio: None,
        };

        Ok(Some(signal_data))
    }

    /// Check if MFI shows overbought condition
    pub fn is_overbought(&self) -> bool {
        if let Some(mfi_value) = self.mfi_history.back() {
            *mfi_value >= self.config.overbought_threshold
        } else {
            false
        }
    }

    /// Check if MFI shows oversold condition
    pub fn is_oversold(&self) -> bool {
        if let Some(mfi_value) = self.mfi_history.back() {
            *mfi_value <= self.config.oversold_threshold
        } else {
            false
        }
    }

    /// Get current MFI value
    pub fn get_mfi_value(&self) -> Option<Decimal> {
        self.mfi_history.back().cloned()
    }

    /// Check if indicator is ready
    pub fn is_ready(&self) -> bool {
        self.is_ready
    }

    /// Reset indicator
    pub fn reset(&mut self) {
        self.typical_prices.clear();
        self.raw_money_flows.clear();
        self.positive_flows.clear();
        self.negative_flows.clear();
        self.price_history.clear();
        self.mfi_history.clear();
        self.timestamps.clear();
        self.is_ready = false;
    }
}

impl Default for MoneyFlowIndex {
    fn default() -> Self {
        Self {
            config: MfiConfig::default(),
            typical_prices: VecDeque::new(),
            raw_money_flows: VecDeque::new(),
            positive_flows: VecDeque::new(),
            negative_flows: VecDeque::new(),
            price_history: VecDeque::new(),
            mfi_history: VecDeque::new(),
            timestamps: VecDeque::new(),
            is_ready: false,
        }
    }
}

/// Helper enum for trend direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrendDir {
    Up,
    Down,
    Sideways,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_mfi_creation() {
        let config = MfiConfig::default();
        let mfi = MoneyFlowIndex::new(config).unwrap();
        assert!(!mfi.is_ready());
    }

    #[test]
    fn test_mfi_calculation() {
        let mut mfi = MoneyFlowIndex::new(MfiConfig {
            period: 3,
            ..Default::default()
        })
        .unwrap();

        // Add test data
        for i in 0..5 {
            let ohlc = OhlcData {
                open: dec!(100),
                high: dec!(105) + Decimal::from(i),
                low: dec!(95) + Decimal::from(i),
                close: dec!(100) + Decimal::from(i),
                volume: dec!(1000),
                timestamp: Utc::now(),
                quote_volume: Some(dec!(100000) + Decimal::from(i * 1000)),
                trade_count: Some(100),
                taker_buy_base_volume: Some(dec!(600)),
                taker_buy_quote_volume: Some(dec!(60000) + Decimal::from(i * 600)),
            };

            let result = mfi.update(&ohlc).unwrap();
            if i >= 3 {
                assert!(result.is_some());
                let mfi_value = result.unwrap().value;
                assert!(mfi_value >= dec!(0) && mfi_value <= dec!(100));
            }
        }

        assert!(mfi.is_ready());
    }

    #[test]
    fn test_mfi_overbought_oversold() {
        let mut mfi = MoneyFlowIndex::new(MfiConfig {
            period: 3,
            overbought_threshold: dec!(80),
            oversold_threshold: dec!(20),
            ..Default::default()
        })
        .unwrap();

        // Add enough data to make it ready - use varying prices to get more realistic MFI
        let test_data = vec![
            (dec!(100), dec!(105), dec!(95), dec!(100), dec!(1000)),
            (dec!(100), dec!(104), dec!(96), dec!(99), dec!(800)),
            (dec!(99), dec!(103), dec!(97), dec!(98), dec!(900)),
            (dec!(98), dec!(102), dec!(96), dec!(97), dec!(1100)),
            (dec!(97), dec!(101), dec!(95), dec!(96), dec!(1200)),
        ];

        for (open, high, low, close, volume) in test_data {
            let ohlc = OhlcData {
                open,
                high,
                low,
                close,
                volume,
                timestamp: Utc::now(),
                quote_volume: Some(volume * close),
                trade_count: Some(100),
                taker_buy_base_volume: Some(volume * dec!(0.6)),
                taker_buy_quote_volume: Some(volume * close * dec!(0.6)),
            };
            mfi.update(&ohlc).unwrap();
        }

        assert!(mfi.is_ready());
        // With declining prices, MFI should be below 50 (bearish)
        // So it should not be overbought
        assert!(!mfi.is_overbought());

        // Test overbought condition with rising prices
        let mut mfi_bullish = MoneyFlowIndex::new(MfiConfig {
            period: 3,
            overbought_threshold: dec!(70), // Lower threshold for easier testing
            oversold_threshold: dec!(30),
            ..Default::default()
        })
        .unwrap();

        // Add bullish data
        let bullish_data = vec![
            (dec!(100), dec!(105), dec!(98), dec!(104), dec!(2000)),
            (dec!(104), dec!(108), dec!(102), dec!(107), dec!(2500)),
            (dec!(107), dec!(111), dec!(105), dec!(110), dec!(3000)),
            (dec!(110), dec!(115), dec!(108), dec!(114), dec!(3500)),
        ];

        for (open, high, low, close, volume) in bullish_data {
            let ohlc = OhlcData {
                open,
                high,
                low,
                close,
                volume,
                timestamp: Utc::now(),
                quote_volume: Some(volume * close),
                trade_count: Some(100),
                taker_buy_base_volume: Some(volume * dec!(0.8)), // High buying pressure
                taker_buy_quote_volume: Some(volume * close * dec!(0.8)),
            };
            mfi_bullish.update(&ohlc).unwrap();
        }

        // This should show overbought condition with strong uptrend
        if let Some(mfi_value) = mfi_bullish.get_mfi_value() {
            // MFI should be high with strong buying pressure
            assert!(mfi_value > dec!(50));
        }
    }
}
