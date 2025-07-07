# 🚀 Signalizer - Kripto Para Sinyal Analizi Uygulaması

Kullanıcıların kripto para borsalarından alınan anlık fiyat verileri ile özelleştirilmiş indikatörlerin analiz ettiği alım-satım bölgelerini görebileceği bir Android & iOS mobil uygulaması.

## 🎯 Proje Amacı

Gerçek zamanlı kripto para fiyat verileri kullanarak:
- Özelleştirilmiş indikatör analizleri (RIBQA, GCR fractal, Ricci Flow)
- Alım bölgesi, hedef fiyatlar, stop-loss önerileri
- Push notification ile sinyal duyuruları
- Kullanıcı favori coin takibi

## 🏗️ Teknoloji Yığını

### Backend
- **Framework:** Go (Gin)
- **Database:** Supabase (PostgreSQL)
- **Cache:** Redis
- **Real-time:** Supabase Realtime
- **Authentication:** Supabase Auth

### Mobile
- **Framework:** Flutter (Dart)
- **State Management:** BLoC
- **Database:** Supabase Flutter SDK
- **Notifications:** Firebase Cloud Messaging

### External APIs
- **Binance API:** Real-time price data
- **CoinGecko API:** Market data
- **WebSocket:** Live price updates

## 📁 Proje Yapısı

```
signalizer/
├── backend/                    # Go Backend API
│   ├── cmd/api/               # Application entry point
│   ├── internal/              # Private application code
│   │   ├── config/           # Configuration
│   │   ├── domain/           # Domain entities
│   │   ├── repository/       # Data access layer
│   │   ├── usecase/          # Business logic
│   │   ├── delivery/         # HTTP handlers
│   │   └── services/         # External services
│   ├── pkg/                  # Public packages
│   └── supabase/             # Database migrations
├── mobile/                    # Flutter Mobile App
├── shared/                    # Shared resources
├── docs/                      # Documentation
├── scripts/                   # Deployment scripts
└── docker/                    # Docker configurations
```

## 🚀 Kurulum ve Çalıştırma

### Gereksinimler
- Go 1.21+
- Flutter 3.0+
- Redis
- Supabase hesabı

### Backend Setup

1. **Dependencies yükle:**
```bash
cd backend
go mod download
```

2. **Environment variables ayarla:**
```bash
cp env.example .env
# .env dosyasını düzenle
```

3. **Supabase migrations çalıştır:**
```bash
# Supabase CLI ile
supabase db push
```

4. **Backend'i çalıştır:**
```bash
go run cmd/api/main.go
```

### Mobile Setup

1. **Dependencies yükle:**
```bash
cd mobile
flutter pub get
```

2. **iOS için:**
```bash
cd ios && pod install
```

3. **Uygulamayı çalıştır:**
```bash
flutter run
```

## 🔧 Geliştirme

### Backend API Endpoints

```
GET    /api/v1/coins              # Coin listesi
GET    /api/v1/coins/:id          # Coin detayı
GET    /api/v1/signals            # Sinyal listesi
POST   /api/v1/signals            # Yeni sinyal
GET    /api/v1/user/favorites     # Kullanıcı favorileri
POST   /api/v1/user/favorites     # Favori ekle
DELETE /api/v1/user/favorites/:id # Favori sil
```

### Database Schema

#### Coins Table
```sql
CREATE TABLE coins (
    id UUID PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    name VARCHAR(100) NOT NULL,
    current_price DECIMAL(20,8),
    market_cap BIGINT,
    volume_24h BIGINT,
    price_change_24h DECIMAL(10,4),
    last_updated TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true
);
```

#### Signals Table
```sql
CREATE TABLE signals (
    id UUID PRIMARY KEY,
    coin_id UUID REFERENCES coins(id),
    signal_type VARCHAR(20) NOT NULL,
    buy_zone_min DECIMAL(20,8),
    buy_zone_max DECIMAL(20,8),
    target_1 DECIMAL(20,8),
    target_2 DECIMAL(20,8),
    target_3 DECIMAL(20,8),
    stop_loss DECIMAL(20,8),
    confidence_score DECIMAL(3,2),
    indicator_used VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true
);
```

## 🔒 Güvenlik

- **Row Level Security (RLS):** Supabase ile kullanıcı bazlı veri erişimi
- **JWT Authentication:** Supabase Auth ile otomatik token yönetimi
- **API Rate Limiting:** DDoS koruması
- **Input Validation:** Tüm API endpoint'lerinde

## 📊 Özellikler

### ✅ Tamamlanan
- [x] Backend temel yapısı
- [x] Supabase entegrasyonu
- [x] Domain modelleri
- [x] Database migrations
- [x] Configuration management

### 🔄 Devam Eden
- [ ] API endpoints
- [ ] Flutter mobil app
- [ ] Binance API entegrasyonu
- [ ] Signal generation engine

### 📋 Planlanan
- [ ] Push notifications
- [ ] Real-time updates
- [ ] Advanced indicators
- [ ] Performance analytics

## 🤝 Katkıda Bulunma

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 📞 İletişim

- **Proje Sahibi:** Genshro
- **Repository:** [https://github.com/Genshro/signalizer](https://github.com/Genshro/signalizer)

---

**Not:** Bu proje aktif geliştirme aşamasındadır. Production kullanımı için stable release'leri bekleyin. 