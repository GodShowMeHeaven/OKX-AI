import asyncio
import websockets
import json
import hmac
import hashlib
import base64
import time
import requests
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import openai
from dataclasses import dataclass
from dotenv import load_dotenv
import os
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    action: str  # 'long', 'short', 'close', 'hold'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_percent: float
    reasoning: str

class OKXFuturesBot:
    def __init__(self, api_key: str, secret_key: str, passphrase: str, openai_key: str):
        # OKX API credentials
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        
        # OpenAI setup
        self.openai_client = openai.Client(api_key=openai_key)  # Новый клиент OpenAI
        
        # Trading settings
        self.symbol = "ETH-USDT-SWAP"
        self.leverage = 20
        
        # Data storage
        self.candles_1m = []
        self.candles_5m = []
        self.funding_rate = None
        self.open_interest = None
        self.long_short_ratio = None
        self.orderbook = None
        
        # WebSocket URLs
        self.ws_public_url = "wss://ws.okx.com:8443/ws/v5/public"
        self.ws_private_url = "wss://ws.okx.com:8443/ws/v5/private"
        
        # REST API base
        self.base_url = "https://www.okx.com"
        
        # Trading state
        self.current_position = None
        self.last_analysis_time = 0
        
    def generate_signature(self, timestamp: str, method: str, request_path: str, body: str = ""):
        """Generate OKX API signature"""
        message = timestamp + method + request_path + body
        mac = hmac.new(
            bytes(self.secret_key, encoding='utf8'),
            bytes(message, encoding='utf-8'),
            digestmod=hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode()
    
    def get_headers(self, method: str, request_path: str, body: str = ""):
        """Generate headers for OKX API requests"""
        timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
        signature = self.generate_signature(timestamp, method, request_path, body)
        
        return {
            'OK-ACCESS-KEY': self.api_key,
            'OK-ACCESS-SIGN': signature,
            'OK-ACCESS-TIMESTAMP': timestamp,
            'OK-ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
    
    async def get_historical_data(self):
        """Get initial historical data via REST API"""
        try:
            # Get 1m candles
            url_1m = f"{self.base_url}/api/v5/market/history-candles"
            params_1m = {
                'instId': self.symbol,
                'bar': '1m',
                'limit': '200'
            }
            
            response_1m = requests.get(url_1m, params=params_1m)
            if response_1m.status_code == 200:
                data_1m = response_1m.json()
                if data_1m['code'] == '0':
                    self.candles_1m = data_1m['data'][::-1]  # Reverse to get chronological order
                    logger.info(f"Loaded {len(self.candles_1m)} 1m candles")
            
            # Get 5m candles
            params_5m = {
                'instId': self.symbol,
                'bar': '5m',
                'limit': '100'
            }
            
            response_5m = requests.get(url_1m, params=params_5m)
            if response_5m.status_code == 200:
                data_5m = response_5m.json()
                if data_5m['code'] == '0':
                    self.candles_5m = data_5m['data'][::-1]
                    logger.info(f"Loaded {len(self.candles_5m)} 5m candles")
                    
            # Get funding rate
            funding_url = f"{self.base_url}/api/v5/public/funding-rate"
            funding_params = {'instId': self.symbol}
            funding_response = requests.get(funding_url, params=funding_params)
            if funding_response.status_code == 200:
                funding_data = funding_response.json()
                if funding_data['code'] == '0' and funding_data['data']:
                    self.funding_rate = float(funding_data['data'][0]['fundingRate'])
                    logger.info(f"Current funding rate: {self.funding_rate}")
                    
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
    
    def calculate_indicators(self) -> Dict:
        """Calculate technical indicators"""
        if not self.candles_1m or len(self.candles_1m) < 50:
            return {}
        
        # Convert to pandas DataFrame for easier calculation
        df = pd.DataFrame(self.candles_1m, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
        df['close'] = df['c'].astype(float)
        df['high'] = df['h'].astype(float)
        df['low'] = df['l'].astype(float)
        df['volume'] = df['vol'].astype(float)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        # Защита от деления на ноль
        rs = gain / (loss + 1e-10)  # Добавляем малую константу
        rsi = 100 - (100 / (1 + rs))
        
        # EMA
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        
        # MACD
        ema_12 = df['close'].ewm(span=12).mean()
        ema_26 = df['close'].ewm(span=26).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9).mean()
        macd_histogram = macd_line - signal_line
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = df['close'].rolling(window=bb_period).mean()
        bb_std_val = df['close'].rolling(window=bb_period).std()
        bb_upper = bb_middle + (bb_std_val * bb_std)
        bb_lower = bb_middle - (bb_std_val * bb_std)
        
        return {
            'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
            'ema_20': ema_20.iloc[-1],
            'ema_50': ema_50.iloc[-1],
            'macd': macd_line.iloc[-1],
            'macd_signal': signal_line.iloc[-1],
            'macd_histogram': macd_histogram.iloc[-1],
            'bb_upper': bb_upper.iloc[-1],
            'bb_middle': bb_middle.iloc[-1],
            'bb_lower': bb_lower.iloc[-1],
            'current_price': df['close'].iloc[-1],
            'volume_avg': df['volume'].tail(20).mean()
        }
    
    async def check_market_gaps(self) -> bool:
        """Проверяет наличие рыночных гэпов на основе последних свечей."""
        try:
            if len(self.candles_1m) < 10:
                logger.warning("Недостаточно данных для анализа гэпов")
                return False

            # Конвертируем свечи в DataFrame
            df = pd.DataFrame(self.candles_1m, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
            df['close'] = df['c'].astype(float)

            # Рассчитываем процентное изменение цены за последнюю минуту
            last_candle = df['close'].iloc[-1]
            prev_candle = df['close'].iloc[-2]
            price_change_percent = abs((last_candle - prev_candle) / prev_candle) * 100

            # Порог для гэпа (например, 1%)
            gap_threshold = 1.0
            if price_change_percent > gap_threshold:
                logger.warning(f"Обнаружен рыночный гэп: {price_change_percent:.2f}% > {gap_threshold}%")
                return True

            # Проверка волатильности через ATR (Average True Range)
            df['tr'] = np.maximum(df['h'].astype(float) - df['l'].astype(float),
                                  np.maximum(abs(df['h'].astype(float) - df['c'].shift().astype(float)),
                                            abs(df['l'].astype(float) - df['c'].shift().astype(float))))
            atr = df['tr'].rolling(window=14).mean().iloc[-1]
            atr_percent = (atr / last_candle) * 100

            # Порог волатильности (например, 0.5% от текущей цены)
            volatility_threshold = 0.5
            if atr_percent > volatility_threshold:
                logger.warning(f"Высокая волатильность: ATR {atr_percent:.2f}% > {volatility_threshold}%")
                return True

            return False

        except Exception as e:
            logger.error(f"Ошибка при проверке гэпов: {e}")
            return False
    
    async def check_liquidity(self, entry_price: float, position_size: float, side: str) -> bool:
        """Проверяет ликвидность в ордербуке для входа в позицию."""
        try:
            if not self.orderbook:
                logger.warning("Данные ордербука недоступны")
                return False

            # Извлекаем bid/ask из ордербука
            bids = self.orderbook.get('bids', [])  # [[price, size], ...]
            asks = self.orderbook.get('asks', [])  # [[price, size], ...]

            # Рассчитываем необходимый объем в USDT
            required_volume = position_size * entry_price

            # Проверяем ликвидность в зависимости от стороны позиции
            if side == "buy":
                # Для покупки проверяем asks (продажи)
                available_volume = sum(float(ask[1]) * float(ask[0]) for ask in asks if float(ask[0]) <= entry_price * 1.005)
                if available_volume < required_volume:
                    logger.warning(f"Недостаточная ликвидность для покупки: доступно ${available_volume:.2f}, требуется ${required_volume:.2f}")
                    return False
            else:  # side == "sell"
                # Для продажи проверяем bids (покупки)
                available_volume = sum(float(bid[1]) * float(bid[0]) for bid in bids if float(bid[0]) >= entry_price * 0.995)
                if available_volume < required_volume:
                    logger.warning(f"Недостаточная ликвидность для продажи: доступно ${available_volume:.2f}, требуется ${required_volume:.2f}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Ошибка при проверке ликвидности: {e}")
            return False
    
    async def analyze_with_openai(self) -> Optional[TradingSignal]:
        """Send data to OpenAI for analysis"""
        try:
            indicators = self.calculate_indicators()
            if not indicators:
                return None
            
            # Prepare market data summary
            recent_candles = self.candles_1m[-10:] if len(self.candles_1m) >= 10 else self.candles_1m
            
            prompt = f"""
Ты опытный трейдер фьючерсов ETH-USDT с плечом {self.leverage}x. Проанализируй следующие данные и дай торговую рекомендацию:

ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:
- RSI(14): {indicators.get('rsi', 0):.2f}
- EMA(20): {indicators.get('ema_20', 0):.2f}
- EMA(50): {indicators.get('ema_50', 0):.2f}
- MACD: {indicators.get('macd', 0):.4f}
- MACD Signal: {indicators.get('macd_signal', 0):.4f}
- MACD Histogram: {indicators.get('macd_histogram', 0):.4f}
- Bollinger Upper: {indicators.get('bb_upper', 0):.2f}
- Bollinger Middle: {indicators.get('bb_middle', 0):.2f}
- Bollinger Lower: {indicators.get('bb_lower', 0):.2f}

РЫНОЧНЫЕ ДАННЫЕ:
- Текущая цена: {indicators.get('current_price', 0):.2f}
- Funding Rate: {self.funding_rate or 0:.6f}
- Средний объем (20 периодов): {indicators.get('volume_avg', 0):.2f}

ПОСЛЕДНИЕ СВЕЧИ (1m):
{json.dumps(recent_candles[-5:], indent=2)}

Верни ТОЛЬКО JSON в следующем формате:
{{
    "action": "long|short|close|hold",
    "confidence": 0.0-1.0,
    "entry_price": цена_входа,
    "stop_loss": цена_стопа,
    "take_profit": цена_профита,
    "position_size_percent": процент_от_депозита(1-20),
    "reasoning": "краткое_обоснование"
}}

ВАЖНО: 
- При плече {self.leverage}x используй строгий риск-менеджмент
- Stop loss не более 0.5% от entry price
- Take profit 1-3% от entry price
- Размер позиции не более 10% депозита при низкой уверенности
- Учитывай funding rate для длинных позиций
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            ai_response = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                signal_data = json.loads(ai_response)
                signal = TradingSignal(
                    action=signal_data.get('action', 'hold'),
                    confidence=signal_data.get('confidence', 0.0),
                    entry_price=signal_data.get('entry_price', indicators.get('current_price', 0)),
                    stop_loss=signal_data.get('stop_loss', 0),
                    take_profit=signal_data.get('take_profit', 0),
                    position_size_percent=signal_data.get('position_size_percent', 1),
                    reasoning=signal_data.get('reasoning', 'No reasoning provided')
                )
                return signal
            except json.JSONDecodeError:
                logger.error(f"Failed to parse OpenAI response: {ai_response}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI analysis error: {e}")
            return None
    
    async def get_account_balance(self):
        """Get account balance"""
        try:
            request_path = "/api/v5/account/balance"
            headers = self.get_headers("GET", request_path)
            
            response = requests.get(self.base_url + request_path, headers=headers)
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Balance API response: {data}")  # Добавлено для отладки
                if data['code'] == '0' and data['data']:
                    for balance in data['data'][0]['details']:
                        if balance['ccy'] == 'USDT':
                            return float(balance['availBal'])
                    logger.warning(f"No USDT balance found in Trading Account: {data}")
                else:
                    logger.error(f"Invalid balance response: {data}")
            else:
                logger.error(f"HTTP Error: {response.status_code}")
            return 0
        except Exception as e:
            logger.error(f"Error getting balance: {e}")
            return 0
    
    async def set_leverage(self, leverage: int):
        """Set leverage for the trading pair"""
        try:
            request_path = "/api/v5/account/set-leverage"
            body = json.dumps({
                "instId": self.symbol,
                "lever": str(leverage),
                "mgnMode": "cross"  # cross margin mode
            })
            
            headers = self.get_headers("POST", request_path, body)
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0':
                    logger.info(f"Leverage set to {leverage}x successfully")
                    return True
                else:
                    logger.error(f"Failed to set leverage: {data.get('msg', 'Unknown error')}")
            return False
            
        except Exception as e:
            logger.error(f"Error setting leverage: {e}")
            return False
    
    async def get_position_info(self):
        """Get current position information"""
        try:
            request_path = "/api/v5/account/positions"
            params = f"?instId={self.symbol}"
            
            headers = self.get_headers("GET", request_path + params)
            
            response = requests.get(
                self.base_url + request_path + params,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    for position in data['data']:
                        if position['instId'] == self.symbol:
                            pos_size = float(position['pos'])
                            if pos_size != 0:
                                return {
                                    'side': position['posSide'],
                                    'size': abs(pos_size),
                                    'avg_price': float(position['avgPx']),
                                    'unrealized_pnl': float(position['upl']),
                                    'mark_price': float(position['markPx'])
                                }
            return None
            
        except Exception as e:
            logger.error(f"Error getting position info: {e}")
            return None
    
    async def calculate_position_size(self, signal: TradingSignal):
        """Calculate position size based on balance and risk percentage"""
        try:
            balance = await self.get_account_balance()
            if balance <= 0:
                logger.warning("Insufficient balance")
                return 0
            
            # Calculate USD amount to trade
            risk_amount = balance * (signal.position_size_percent / 100)
            
            # With leverage, we can trade more
            position_value = risk_amount * self.leverage
            
            # Convert to ETH size (OKX futures are quoted in ETH)
            position_size = position_value / signal.entry_price
            
            # Check minimum order size (0.01 ETH for ETH-USDT-SWAP)
            min_order_size = 0.01
            if position_size < min_order_size:
                logger.warning(f"Position size {position_size:.3f} ETH is below minimum {min_order_size} ETH")
                return 0
            
            # Round to appropriate precision
            position_size = round(position_size, 3)
            
            logger.info(f"Balance: ${balance:.2f}")
            logger.info(f"Risk amount: ${risk_amount:.2f}")
            logger.info(f"Position value: ${position_value:.2f}")
            logger.info(f"Position size: {position_size} ETH")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
            
    
    async def place_market_order(self, side: str, size: float, reduce_only: bool = False):
        """Place market order"""
        try:
            request_path = "/api/v5/trade/order"
            
            order_data = {
                "instId": self.symbol,
                "tdMode": "cross",  # cross margin
                "side": side,  # "buy" for long, "sell" for short
                "ordType": "market",
                "sz": str(size),
            }
            
            if reduce_only:
                order_data["reduceOnly"] = "true"
            
            body = json.dumps(order_data)
            headers = self.get_headers("POST", request_path, body)
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    order_info = data['data'][0]
                    logger.info(f"Order placed successfully: {order_info['ordId']}")
                    return order_info['ordId']
                else:
                    logger.error(f"Order failed: {data.get('msg', 'Unknown error')}")
            else:
                logger.error(f"HTTP Error: {response.status_code}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None
    
    async def place_limit_order(self, side: str, size: float, price: float, reduce_only: bool = False):
        """Размещает лимитный ордер для минимизации проскальзывания."""
        try:
            request_path = "/api/v5/trade/order"
            
            order_data = {
                "instId": self.symbol,
                "tdMode": "cross",
                "side": side,
                "ordType": "limit",
                "sz": str(size),
                "px": str(price),  # Цена лимитного ордера
            }
            
            if reduce_only:
                order_data["reduceOnly"] = "true"
            
            body = json.dumps(order_data)
            headers = self.get_headers("POST", request_path, body)
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    order_info = data['data'][0]
                    logger.info(f"Лимитный ордер размещён: {order_info['ordId']} по цене {price}")
                    return order_info['ordId']
                else:
                    logger.error(f"Ошибка размещения лимитного ордера: {data.get('msg', 'Unknown error')}")
            else:
                logger.error(f"HTTP Error: {response.status_code}")
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка при размещении лимитного ордера: {e}")
            return None
    
    async def cancel_order(self, order_id: str):
        """Отменяет ордер по его ID."""
        try:
            request_path = "/api/v5/trade/cancel-order"
            body = json.dumps({
                "instId": self.symbol,
                "ordId": order_id
            })
            
            headers = self.get_headers("POST", request_path, body)
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0':
                    logger.info(f"Order {order_id} cancelled successfully")
                    return True
                else:
                    logger.error(f"Failed to cancel order: {data.get('msg', 'Unknown error')}")
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return False
    
    async def check_slippage(self, order_id: str, expected_price: float) -> bool:
        """Проверяет проскальзывание для ордера."""
        try:
            request_path = f"/api/v5/trade/order?ordId={order_id}"
            headers = self.get_headers("GET", request_path)
            
            response = requests.get(self.base_url + request_path, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    fill_price = float(data['data'][0].get('fillPx', 0))
                    if fill_price == 0:  # Ордер ещё не исполнен
                        return True  # Считаем, что проскальзывания нет, пока ордер не исполнен
                    slippage = abs((fill_price - expected_price) / expected_price) * 100
                    slippage_threshold = 0.5  # Порог проскальзывания 0.5%
                    if slippage > slippage_threshold:
                        logger.warning(f"Проскальзывание {slippage:.2f}% превышает порог {slippage_threshold}%")
                        return False
                    return True
            return False
        except Exception as e:
            logger.error(f"Ошибка при проверке проскальзывания: {e}")
            return False
    
    async def place_stop_loss_take_profit(self, side: str, size: float, stop_loss: float, take_profit: float):
        """Place stop loss and take profit orders"""
        try:
            # Determine order sides for SL and TP
            if side == "buy":  # Long position
                sl_side = "sell"
                tp_side = "sell" 
            else:  # Short position
                sl_side = "buy"
                tp_side = "buy"
            
            orders_placed = []
            
            # Place Stop Loss order
            if stop_loss > 0:
                sl_order_data = {
                    "instId": self.symbol,
                    "tdMode": "cross",
                    "side": sl_side,
                    "ordType": "conditional",
                    "sz": str(size),
                    "triggerPx": str(stop_loss),
                    "orderPx": "-1",  # Market price when triggered
                    "triggerPxType": "mark",  # Use mark price to avoid manipulation
                    "reduceOnly": "true"
                }
                
                sl_response = await self._place_conditional_order(sl_order_data, "Stop Loss")
                if sl_response:
                    orders_placed.append(("SL", sl_response))
            
            # Place Take Profit order
            if take_profit > 0:
                tp_order_data = {
                    "instId": self.symbol,
                    "tdMode": "cross",
                    "side": tp_side,
                    "ordType": "conditional",
                    "sz": str(size),
                    "triggerPx": str(take_profit),
                    "orderPx": "-1",  # Market price when triggered
                    "triggerPxType": "mark",
                    "reduceOnly": "true"
                }
                
                tp_response = await self._place_conditional_order(tp_order_data, "Take Profit")
                if tp_response:
                    orders_placed.append(("TP", tp_response))
            
            return orders_placed
            
        except Exception as e:
            logger.error(f"Error placing SL/TP orders: {e}")
            return []
    
    async def _place_conditional_order(self, order_data: dict, order_type: str):
        """Helper function to place conditional orders"""
        try:
            request_path = "/api/v5/trade/order-algo"
            body = json.dumps(order_data)
            headers = self.get_headers("POST", request_path, body)
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    order_info = data['data'][0]
                    logger.info(f"{order_type} order placed: {order_info['algoId']}")
                    return order_info['algoId']
                else:
                    logger.error(f"{order_type} order failed: {data.get('msg', 'Unknown error')}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error placing {order_type} order: {e}")
            return None
    
    async def close_position(self, reason: str = "Signal"):
        """Close current position"""
        try:
            position = await self.get_position_info()
            if not position:
                logger.info("No position to close")
                return True
            
            # Determine close side
            close_side = "sell" if position['side'] == "long" else "buy"
            
            logger.info(f"Closing position - Reason: {reason}")
            logger.info(f"Position: {position['side']} {position['size']} ETH")
            logger.info(f"Unrealized PnL: ${position['unrealized_pnl']:.2f}")
            
            order_id = await self.place_market_order(
                side=close_side,
                size=position['size'],
                reduce_only=True
            )
            
            if order_id:
                logger.info(f"Position closed successfully")
                # Cancel any remaining SL/TP orders
                await self.cancel_all_algo_orders()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    async def cancel_all_algo_orders(self):
        """Cancel all algorithmic orders (SL/TP)"""
        try:
            request_path = "/api/v5/trade/cancel-algos"
            body = json.dumps([{
                "instId": self.symbol,
                "algoId": ""  # Cancel all for this instrument
            }])
            
            headers = self.get_headers("POST", request_path, body)
            
            response = requests.post(
                self.base_url + request_path,
                headers=headers,
                data=body
            )
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0':
                    logger.info("All algo orders cancelled")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error cancelling algo orders: {e}")
            return False
    
    async def place_order(self, signal: TradingSignal):
        """Place order based on trading signal"""
        logger.info(f"TRADING SIGNAL RECEIVED: Action={signal.action}, Confidence={signal.confidence:.2f}, "
                    f"Entry Price={signal.entry_price:.2f}, Stop Loss={signal.stop_loss:.2f}, "
                    f"Take Profit={signal.take_profit:.2f}, Position Size={signal.position_size_percent}%, "
                    f"Reasoning={signal.reasoning}")
        
        try:
            # Проверка на рыночные гэпы
            if await self.check_market_gaps():
                logger.warning("Торговля приостановлена из-за рыночного гэпа или высокой волатильности")
                return
            
            # Check current position
            current_position = await self.get_position_info()
            
            if signal.action == "hold":
                logger.info("Signal: HOLD - No action taken")
                return
            
            elif signal.action == "close":
                if current_position:
                    await self.close_position("AI Signal")
                else:
                    logger.info("No position to close")
                return
            
            elif signal.action in ["long", "short"]:
                # Close opposite position if exists
                if current_position:
                    current_side = current_position['side']
                    signal_side = "long" if signal.action == "long" else "short"
                    
                    if current_side != signal_side:
                        logger.info(f"Closing opposite position: {current_side}")
                        await self.close_position("Opposite Signal")
                        await asyncio.sleep(2)  # Wait for close to complete
                    else:
                        logger.info(f"Already in {signal_side} position, skipping")
                        return
                
                # Set leverage if not set
                await self.set_leverage(self.leverage)
                await asyncio.sleep(1)
                
                # Calculate position size
                position_size = await self.calculate_position_size(signal)
                if position_size <= 0:
                    logger.warning("Invalid position size, skipping trade")
                    return
                
                # Проверка ликвидности
                side = "buy" if signal.action == "long" else "sell"
                if not await self.check_liquidity(signal.entry_price, position_size, side):
                    logger.warning("Торговля приостановлена из-за недостаточной ликвидности")
                    return
                
                # Place limit order instead of market order to minimize slippage
                order_id = await self.place_limit_order(
                    side=side,
                    size=position_size,
                    price=signal.entry_price
                )
                
                if order_id:
                    # Ожидание исполнения лимитного ордера (максимум 30 секунд)
                    timeout = 30
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        order_status = await self.check_slippage(order_id, signal.entry_price)
                        if order_status:
                            break
                        await asyncio.sleep(2)
                    
                    # Если ордер не исполнен, отменяем его
                    if time.time() - start_time >= timeout:
                        logger.warning(f"Лимитный ордер {order_id} не исполнен за {timeout} секунд, отмена")
                        await self.cancel_order(order_id)
                        return
                    
                    # Проверка проскальзывания
                    if not await self.check_slippage(order_id, signal.entry_price):
                        logger.warning("Закрытие позиции из-за высокого проскальзывания")
                        await self.close_position("High Slippage")
                        return
                    
                    # Place stop loss and take profit
                    if signal.stop_loss > 0 or signal.take_profit > 0:
                        sl_tp_orders = await self.place_stop_loss_take_profit(
                            side=side,
                            size=position_size,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit
                        )
                        
                        logger.info(f"SL/TP orders placed: {len(sl_tp_orders)}")
                    
                    # Update current position info
                    self.current_position = await self.get_position_info()
                    if self.current_position:
                        logger.info(f"New position: {self.current_position}")
                else:
                    logger.error("Failed to place limit order")
        
        except Exception as e:
            logger.error(f"Error in place_order: {e}")
    
    async def handle_websocket_message(self, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if 'data' in data:
                for item in data['data']:
                    # Handle different message types
                    if 'arg' in data and data['arg'].get('channel') == 'candle1m':
                        # Update 1m candles
                        self.candles_1m.append(item)
                        if len(self.candles_1m) > 200:
                            self.candles_1m.pop(0)
                    
                    elif 'arg' in data and data['arg'].get('channel') == 'candle5m':
                        # Update 5m candles
                        self.candles_5m.append(item)
                        if len(self.candles_5m) > 100:
                            self.candles_5m.pop(0)
                    
                    elif 'arg' in data and data['arg'].get('channel') == 'books50':
                        self.orderbook = item  # Сохраняем данные ордербука
                        logger.info(f"Обновлён ордербук: {len(item.get('bids', []))} bids, {len(item.get('asks', []))} asks")
                    
                    elif 'arg' in data and data['arg'].get('channel') == 'funding-rate':
                        self.funding_rate = float(item.get('fundingRate', 0))
                        logger.info(f"Обновлена ставка финансирования: {self.funding_rate:.6f}")
        
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    async def websocket_client(self):
        """WebSocket client для real-time данных с переподключением."""
        reconnect_delay = 5

        while True:
            try:
                async with websockets.connect(self.ws_public_url) as websocket:
                    subscriptions = {
                        "op": "subscribe",
                        "args": [
                            {"channel": "candle1m", "instId": self.symbol},
                            {"channel": "candle5m", "instId": self.symbol},
                            {"channel": "books50", "instId": self.symbol},  # Изменено на books50 для большей глубины
                            {"channel": "funding-rate", "instId": self.symbol}
                        ]
                    }
                    
                    await websocket.send(json.dumps(subscriptions))
                    logger.info("Subscribed to WebSocket channels")
                    
                    async for message in websocket:
                        await self.handle_websocket_message(message)
            
            except (websockets.exceptions.ConnectionClosed, Exception) as e:
                logger.error(f"WebSocket error: {e}. Reconnecting in {reconnect_delay} seconds...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)  # Экспоненциальная задержка, максимум 60 секунд
    
    async def analysis_loop(self):
        """Main analysis loop - runs every 60 seconds"""
        while True:
            try:
                current_time = time.time()
                if current_time - self.last_analysis_time >= 60:  # Analyze every 60 seconds
                    logger.info(f"Running analysis at {datetime.now()}")
                    
                    signal = await self.analyze_with_openai()
                    if signal and signal.confidence >= 0.6:  # Only act on high confidence signals
                        await self.place_order(signal)
                    
                    self.last_analysis_time = current_time
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")
                await asyncio.sleep(10)
    
    async def start(self):
        """Start the trading bot"""
        logger.info("Starting ETH-USDT Futures Trading Bot...")
        
        # Check for open orders
        try:
            request_path = "/api/v5/trade/orders-pending"
            headers = self.get_headers("GET", request_path)
            response = requests.get(self.base_url + request_path, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    logger.warning(f"Found {len(data['data'])} open orders. Cancel them to free up balance.")
            else:
                logger.error(f"Error checking open orders: HTTP {response.status_code}, {response.text}")
        except Exception as e:
            logger.error(f"Error checking open orders: {e}")
        
        # Set initial leverage
        await self.set_leverage(self.leverage)
        
        # Wait for balance synchronization
        await asyncio.sleep(5)
        
        # Check account balance
        balance = await self.get_account_balance()
        logger.info(f"Account balance: ${balance:.2f} USDT")
        
        if balance < 10:  # Изменено с 20 на 10
            logger.warning("Insufficient balance to start trading")
            return
        
        # Load initial historical data
        await self.get_historical_data()
        
        # Check for existing positions
        existing_position = await self.get_position_info()
        if existing_position:
            logger.info(f"Existing position found: {existing_position}")
            self.current_position = existing_position
        
        # Start WebSocket and analysis tasks
        tasks = [
            asyncio.create_task(self.websocket_client()),
            asyncio.create_task(self.analysis_loop()),
            asyncio.create_task(self.position_monitor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Bot error: {e}")
    
    async def position_monitor(self):
        """Monitor position and manage risk"""
        while True:
            try:
                if self.current_position:
                    updated_position = await self.get_position_info()
                    if updated_position:
                        self.current_position = updated_position
                        
                        # Risk management checks
                        unrealized_pnl = updated_position['unrealized_pnl']
                        position_value = updated_position['size'] * updated_position['mark_price']
                        pnl_percent = (unrealized_pnl / position_value) * 100 if position_value > 0 else 0
                        
                        logger.info(f"Position Status - Side: {updated_position['side']}, "
                                    f"Size: {updated_position['size']}, "
                                    f"PnL: ${unrealized_pnl:.2f} ({pnl_percent:.2f}%)")
                        
                        # Emergency stop if loss is too high
                        if pnl_percent < -5:  # 5% loss emergency stop
                            logger.warning("EMERGENCY STOP: Loss exceeded 5%")
                            await self.close_position("Emergency Stop")
                            self.current_position = None
                    else:
                        # Position was closed
                        if self.current_position:
                            logger.info("Position was closed")
                            self.current_position = None
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
                await asyncio.sleep(30)

async def main():
    load_dotenv()  # Для локальной разработки
    api_key = os.getenv("OKX_API_KEY")
    secret_key = os.getenv("OKX_SECRET_KEY")
    passphrase = os.getenv("OKX_PASSPHRASE")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    logger.debug(f"API Key: {api_key[:4] if api_key else None}****")
    logger.debug(f"Secret Key: {secret_key[:4] if secret_key else None}****")
    logger.debug(f"Passphrase: {passphrase[:4] if passphrase else None}****")
    
    if not all([api_key, secret_key, passphrase, openai_key]):
        logger.error("Missing environment variables")
        raise ValueError("One or more environment variables are missing")

    bot = OKXFuturesBot(
        api_key=api_key,
        secret_key=secret_key,
        passphrase=passphrase,
        openai_key=openai_key
    )
    
    await bot.start()

if __name__ == "__main__":
    asyncio.run(main())