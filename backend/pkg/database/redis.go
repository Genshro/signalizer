package database

import (
	"context"
	"time"

	"github.com/redis/go-redis/v9"
)

type RedisClient struct {
	client *redis.Client
	ctx    context.Context
}

func NewRedisClient(url, password string) *RedisClient {
	opt, err := redis.ParseURL(url)
	if err != nil {
		// Fallback to default options
		opt = &redis.Options{
			Addr:     "localhost:6379",
			Password: password,
			DB:       0,
		}
	}

	if password != "" {
		opt.Password = password
	}

	rdb := redis.NewClient(opt)
	ctx := context.Background()

	return &RedisClient{
		client: rdb,
		ctx:    ctx,
	}
}

func (r *RedisClient) Set(key string, value interface{}, expiration time.Duration) error {
	return r.client.Set(r.ctx, key, value, expiration).Err()
}

func (r *RedisClient) Get(key string) (string, error) {
	return r.client.Get(r.ctx, key).Result()
}

func (r *RedisClient) Del(key string) error {
	return r.client.Del(r.ctx, key).Err()
}

func (r *RedisClient) Exists(key string) (bool, error) {
	result, err := r.client.Exists(r.ctx, key).Result()
	return result > 0, err
}

func (r *RedisClient) HSet(key string, field string, value interface{}) error {
	return r.client.HSet(r.ctx, key, field, value).Err()
}

func (r *RedisClient) HGet(key string, field string) (string, error) {
	return r.client.HGet(r.ctx, key, field).Result()
}

func (r *RedisClient) HGetAll(key string) (map[string]string, error) {
	return r.client.HGetAll(r.ctx, key).Result()
}

func (r *RedisClient) Publish(channel string, message interface{}) error {
	return r.client.Publish(r.ctx, channel, message).Err()
}

func (r *RedisClient) Subscribe(channels ...string) *redis.PubSub {
	return r.client.Subscribe(r.ctx, channels...)
}

func (r *RedisClient) ZAdd(key string, members ...*redis.Z) error {
	return r.client.ZAdd(r.ctx, key, members...).Err()
}

func (r *RedisClient) ZRange(key string, start, stop int64) ([]string, error) {
	return r.client.ZRange(r.ctx, key, start, stop).Result()
}

func (r *RedisClient) ZRangeWithScores(key string, start, stop int64) ([]redis.Z, error) {
	return r.client.ZRangeWithScores(r.ctx, key, start, stop).Result()
}

func (r *RedisClient) Ping() error {
	return r.client.Ping(r.ctx).Err()
}

func (r *RedisClient) Close() error {
	return r.client.Close()
}
