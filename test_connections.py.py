import asyncio
import requests
import json
import pandas as pd
import openai
from datetime import datetime
import hmac
import hashlib
import base64
from trading_bot import OKXFuturesBot

class TestBotConnections:
    def __init__(self, api_key, secret_key, passphrase, openai_key):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.openai_key = openai_key
        self.base_url = "https://www.okx.com"
        self.symbol = "ETH-USDT-SWAP"
    
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
    
    def test_okx_public_api(self):
        """Тест публичного API (не требует ключей)"""
        print("🔍 Тестируем публичный API OKX...")
        
        try:
            # Тест получения тикера
            url = f"{self.base_url}/api/v5/market/ticker"
            params = {'instId': self.symbol}
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    ticker = data['data'][0]
                    print(f"✅ Публичный API работает")
                    print(f"   ETH цена: ${float(ticker['last']):.2f}")
                    print(f"   24h объем: {float(ticker['vol24h']):.0f} ETH")
                    return True
            
            print(f"❌ Публичный API ошибка: {response.status_code}")
            return False
            
        except Exception as e:
            print(f"❌ Публичный API исключение: {e}")
            return False
    
    def test_okx_private_api(self):
        """Тест приватного API (требует ключи)"""
        print("\n🔐 Тестируем приватный API OKX...")
        
        try:
            # Тест получения баланса
            request_path = "/api/v5/account/balance"
            headers = self.get_headers("GET", request_path)
            
            response = requests.get(self.base_url + request_path, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0':
                    print("✅ Приватный API работает")
                    
                    if data['data']:
                        for balance in data['data'][0]['details']:
                            if balance['ccy'] == 'USDT' and float(balance['bal']) > 0:
                                print(f"   USDT баланс: ${float(balance['availBal']):.2f}")
                    return True
                else:
                    print(f"❌ API ошибка: {data.get('msg', 'Unknown error')}")
            else:
                print(f"❌ HTTP ошибка: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Приватный API исключение: {e}")
            
        return False
    
    def test_openai_api(self):
        """Тест OpenAI API"""
        print("\n🤖 Тестируем OpenAI API...")
        
        try:
            openai.api_key = self.openai_key
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": "Привет! Ответь просто 'OK' если получил это сообщение."}],
                max_tokens=10
            )
            
            if response.choices[0].message.content:
                print("✅ OpenAI API работает")
                print(f"   Ответ: {response.choices[0].message.content}")
                return True
                
        except Exception as e:
            print(f"❌ OpenAI API ошибка: {e}")
            
        return False
    
    def test_data_analysis(self):
        """Тест получения и анализа данных"""
        print("\n📊 Тестируем анализ данных...")
        
        try:
            # Получаем свечи
            url = f"{self.base_url}/api/v5/market/history-candles"
            params = {
                'instId': self.symbol,
                'bar': '1m',
                'limit': '50'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data['code'] == '0' and data['data']:
                    candles = data['data'][::-1]  # Reverse для хронологического порядка
                    
                    # Конвертируем в DataFrame
                    df = pd.DataFrame(candles, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
                    df['close'] = df['c'].astype(float)
                    
                    # Простой RSI
                    delta = df['close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))
                    
                    print("✅ Анализ данных работает")
                    print(f"   Получено {len(candles)} свечей")
                    print(f"   Текущая цена: ${df['close'].iloc[-1]:.2f}")
                    print(f"   RSI: {rsi.iloc[-1]:.1f}")
                    
                    return True
            
            print("❌ Анализ данных не работает")
            return False
            
        except Exception as e:
            print(f"❌ Анализ данных ошибка: {e}")
            return False
    
    async def test_full_analysis_cycle(self):
        """Полный тест цикла анализа с OpenAI"""
        print("\n🔄 Тестируем полный цикл анализа...")
        
        try:
            # Получаем данные
            url = f"{self.base_url}/api/v5/market/history-candles"
            params = {
                'instId': self.symbol,
                'bar': '1m',
                'limit': '100'
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            candles = data['data'][::-1]
            
            # Рассчитываем индикаторы
            df = pd.DataFrame(candles, columns=['ts', 'o', 'h', 'l', 'c', 'vol', 'volCcy', 'volCcyQuote', 'confirm'])
            df['close'] = df['c'].astype(float)
            
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Отправляем в OpenAI
            prompt = f"""
Ты тестовый трейдер. Проанализируй данные ETH-USDT:
- Текущая цена: {df['close'].iloc[-1]:.2f}
- RSI: {rsi.iloc[-1]:.1f}

Верни JSON в формате:
{{"action": "hold", "confidence": 0.5, "reasoning": "Тестовый анализ"}}
"""

            openai.api_key = self.openai_key
            ai_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200
            )
            
            response_text = ai_response.choices[0].message.content
            
            try:
                signal = json.loads(response_text)
                print("✅ Полный цикл анализа работает")
                print(f"   AI ответ: {signal}")
                return True
            except:
                print("⚠️  AI ответил, но не в JSON формате:")
                print(f"   Ответ: {response_text}")
                return False
                
        except Exception as e:
            print(f"❌ Полный цикл ошибка: {e}")
            return False

    def run_all_tests(self):
        """Запустить все тесты"""
        print("🚀 Запускаем тесты подключений...\n")
        
        results = []
        results.append(("Публичный API", self.test_okx_public_api()))
        results.append(("Приватный API", self.test_okx_private_api()))
        results.append(("OpenAI API", self.test_openai_api()))
        results.append(("Анализ данных", self.test_data_analysis()))
        
        # Асинхронный тест
        loop = asyncio.get_event_loop()
        results.append(("Полный цикл", loop.run_until_complete(self.test_full_analysis_cycle())))
        
        print(f"\n📋 Результаты тестов:")
        print("="*40)
        all_passed = True
        for test_name, passed in results:
            status = "✅ ПРОШЕЛ" if passed else "❌ НЕ ПРОШЕЛ"
            print(f"{test_name:<20} {status}")
            if not passed:
                all_passed = False
        
        print("="*40)
        if all_passed:
            print("🎉 Все тесты прошли! Можно запускать бота.")
        else:
            print("⚠️  Некоторые тесты не прошли. Нужно исправить проблемы.")

# Использование:
if __name__ == "__main__":
    # Замените на ваши ключи
    tester = TestBotConnections(
        api_key="4414458c-0159-4a4b-8451-a067b7d17dcc",
        secret_key="4414458c-0159-4a4b-8451-a067b7d17dcc", 
        passphrase="mktN46RPkdV@cwN",
        openai_key="k-proj-_9n2Y9EsiJ2G0e3ZlDIEcJvek-yxSWa_bLEUDR0oVFV_Fbxw3hEQYQ8GLJ03y4YCoH_fV4XMbcT3BlbkFJmBAr6xMOgpa48DFkt6_JqmODJ_G62WrMcP3R9mSU80h0ssVkeja188cw6z9swbwfnhfqtfbUkA"
    )
    
    tester.run_all_tests()