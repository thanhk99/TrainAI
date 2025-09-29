import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import re
import requests
import os
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.products = []
        
    def load_data(self):
        """Load dữ liệu từ file JSON"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                self.products = data
                print(f"✅ Đã load {len(self.products)} sản phẩm")
                return self.products, []
            else:
                raise ValueError("Cấu trúc JSON không hợp lệ")
                
        except Exception as e:
            print(f"❌ Lỗi load data: {e}")
            return [], []

import requests
import re
import json
import os
from typing import Dict

class GPTQueryAnalyzer:
    def __init__(self):
        # KHẨN CẤP: XÓA API KEY KHỎI CODE!
        # Sử dụng các cách bảo mật bên dưới
        self.api_key = self._get_api_key_safely()
        self.base_url = "https://api.openai.com/v1/chat/completions"
    
    def _get_api_key_safely(self):
        """Lấy API key an toàn - KHÔNG để trong code"""
        # Ưu tiên 1: Environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if api_key:
            return api_key
            
        # Ưu tiên 2: File .env
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                return api_key
        except ImportError:
            pass
            
        # Ưu tiên 3: File config.json
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                api_key = config.get('OPENAI_API_KEY')
                if api_key:
                    return api_key
        except:
            pass
            
        # Ưu tiên 4: Nhập từ người dùng
        print("🔑 Không tìm thấy OpenAI API Key")
        api_key = input("Nhập API key của bạn (bắt đầu với sk-): ").strip()
        
        if api_key and api_key.startswith('sk-'):
            self._save_api_key(api_key)
            return api_key
        else:
            print("⚠️ API Key không hợp lệ. Sử dụng fallback mode.")
            return None
    
    def _save_api_key(self, api_key):
        """Lưu API key vào file .env"""
        try:
            with open('.env', 'w') as f:
                f.write(f'OPENAI_API_KEY={api_key}\n')
            print("✅ Đã lưu API key vào file .env")
            
            # Tạo file .gitignore để tránh commit nhầm
            if not os.path.exists('.gitignore'):
                with open('.gitignore', 'w') as f:
                    f.write('.env\nconfig.json\n__pycache__/\n*.pyc\n')
        except Exception as e:
            print(f"⚠️ Không thể lưu API key: {e}")

    def analyze_query(self, user_query: str) -> Dict:
        """Phân tích query với GPT"""
        if not self.api_key:
            print("🔧 Sử dụng fallback mode (không có API key)")
            return self._fallback_analysis(user_query)
            
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        system_prompt = """Bạn là trợ lý AI cho siêu thị. Phân tích query và trả về JSON:
        {
            "original_query": "Câu hỏi gốc",
            "processed_query": "Từ khóa tìm kiếm",
            "intent": "search | recommendation | compare | budget",
            "category": "Danh mục sản phẩm",
            "brand": "Thương hiệu",
            "attributes": {
                "price_range": "low | medium | high | any",
                "purpose": "cooking | drinking | cleaning | personal_care | gift"
            }
        }"""
        
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "temperature": 0.1,
            "max_tokens": 150
        }
        
        try:
            print("🔄 Đang phân tích với GPT...")
            response = requests.post(
                self.base_url, 
                headers=headers, 
                json=payload, 
                timeout=30  # Tăng timeout
            )
            
            # Xử lý các mã lỗi phổ biến
            if response.status_code == 401:
                print("❌ API Key không hợp lệ hoặc đã hết hạn")
                return self._fallback_analysis(user_query)
            elif response.status_code == 429:
                print("❌ Quá giới hạn rate limit - thử lại sau 60 giây")
                return self._fallback_analysis(user_query)
            elif response.status_code == 403:
                print("❌ Tài khoản bị hạn chế hoặc hết credit")
                return self._fallback_analysis(user_query)
            elif response.status_code == 500:
                print("❌ Lỗi server OpenAI - thử lại sau")
                return self._fallback_analysis(user_query)
            elif response.status_code != 200:
                print(f"❌ Lỗi API: {response.status_code} - {response.text}")
                return self._fallback_analysis(user_query)
                
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse JSON từ response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                analysis = json.loads(json_match.group())
                
                # Tính cost ước tính
                usage = result.get('usage', {})
                total_tokens = usage.get('total_tokens', 0)
                cost = (total_tokens / 1000) * 0.002  # ~$0.002 per 1K tokens
                print(f"💰 Token usage: {total_tokens} (~${cost:.6f})")
                
                return self._validate_analysis(analysis, user_query)
            else:
                print("❌ Không thể parse JSON từ GPT response")
                return self._fallback_analysis(user_query)
                
        except requests.exceptions.Timeout:
            print("❌ GPT request timeout sau 30 giây")
            return self._fallback_analysis(user_query)
        except requests.exceptions.ConnectionError:
            print("❌ Lỗi kết nối - kiểm tra internet")
            return self._fallback_analysis(user_query)
        except requests.exceptions.RequestException as e:
            print(f"❌ Lỗi request: {e}")
            return self._fallback_analysis(user_query)
        except Exception as e:
            print(f"❌ Lỗi không xác định: {e}")
            return self._fallback_analysis(user_query)
    
    def _validate_analysis(self, analysis: Dict, original_query: str) -> Dict:
        """Validate analysis result"""
        # Đảm bảo processed_query không rỗng
        if not analysis.get('processed_query') or analysis['processed_query'].strip() == '':
            analysis['processed_query'] = self._simple_preprocess(original_query)
        
        # Đảm bảo các trường cần thiết tồn tại
        default_analysis = {
            "original_query": original_query,
            "processed_query": self._simple_preprocess(original_query),
            "intent": "search",
            "category": None,
            "brand": None,
            "attributes": {
                "price_range": "any",
                "purpose": "daily_use"
            }
        }
        
        # Merge với giá trị mặc định
        for key, default_value in default_analysis.items():
            if key not in analysis:
                analysis[key] = default_value
            elif isinstance(default_value, dict) and isinstance(analysis[key], dict):
                # Merge nested dictionaries
                for sub_key, sub_default in default_value.items():
                    if sub_key not in analysis[key]:
                        analysis[key][sub_key] = sub_default
        
        return analysis
    
    def _fallback_analysis(self, user_query: str) -> Dict:
        """Fallback analysis khi GPT không hoạt động"""
        query_lower = user_query.lower()
        
        # Phân tích intent đơn giản
        intent = "search"
        if any(word in query_lower for word in ['gợi ý', 'nên mua', 'khuyên', 'recommend']):
            intent = "recommendation"
        elif any(word in query_lower for word in ['so sánh', 'compare']):
            intent = "compare"
        elif any(word in query_lower for word in ['rẻ', 'giá', 'tiết kiệm', 'budget']):
            intent = "budget"
        
        # Phân tích category đơn giản
        category = None
        category_keywords = {
            "Thực phẩm": ['gạo', 'cơm', 'mì', 'bún', 'phở', 'thịt', 'cá', 'rau', 'trái cây'],
            "Đồ uống": ['nước', 'bia', 'rượu', 'sữa', 'cafe', 'trà', 'nước ngọt'],
            "Gia dụng": ['bàn', 'ghế', 'tủ', 'bếp', 'đèn', 'chén', 'dĩa', 'bát'],
            "Vệ sinh": ['giặt', 'tẩy', 'xà phòng', 'dầu gội', 'sữa tắm', 'nước rửa'],
            "Điện tử": ['điện thoại', 'tivi', 'tủ lạnh', 'máy giặt']
        }
        
        for cat, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                category = cat
                break
        
        # Phân tích brand đơn giản
        brand = None
        common_brands = ['vinamilk', 'cocacola', 'pepsi', 'tiger', 'heineken', 'panasonic']
        for brand_name in common_brands:
            if brand_name in query_lower:
                brand = brand_name
                break
        
        # Phân tích price range
        price_range = "any"
        if 'rẻ' in query_lower or 'giá rẻ' in query_lower:
            price_range = "low"
        elif 'đắt' in query_lower or 'cao cấp' in query_lower:
            price_range = "high"
        
        return {
            "original_query": user_query,
            "processed_query": self._simple_preprocess(user_query),
            "intent": intent,
            "category": category,
            "brand": brand,
            "attributes": {
                "price_range": price_range,
                "purpose": "daily_use"
            }
        }
    
    def _simple_preprocess(self, query: str) -> str:
        """Tiền xử lý query đơn giản"""
        if not query:
            return ""
            
        stopwords = ['tôi', 'muốn', 'mua', 'cần', 'tìm', 'có', 'nào', 'ạ', 'ơi', 'cho']
        query = re.sub(r'[^\w\s]', ' ', query.lower())
        words = [word for word in query.split() if word not in stopwords and len(word) > 1]
        return ' '.join(words)
class SmartProductFilter:
    def __init__(self, products):
        self.products = products
        
    def apply_filters(self, products: List[Dict], analysis: Dict) -> List[Dict]:
        """Áp dụng các bộ lọc thông minh"""
        filtered_products = products.copy()
        
        # Lọc theo category
        if analysis["category"]:
            filtered_products = self._filter_by_category(filtered_products, analysis["category"])
        
        # Lọc theo brand
        if analysis["brand"]:
            filtered_products = self._filter_by_brand(filtered_products, analysis["brand"])
        
        # Lọc theo price range
        price_range = analysis["attributes"]["price_range"]
        if price_range != "any":
            filtered_products = self._filter_by_price(filtered_products, price_range)
        
        return filtered_products
    
    def _filter_by_category(self, products: List[Dict], category: str) -> List[Dict]:
        """Lọc sản phẩm theo category"""
        category_lower = category.lower()
        filtered = []
        for product in products:
            product_category = product.get('category', '').lower()
            product_name = product.get('name', '').lower()
            
            if (category_lower in product_category or 
                category_lower in product_name):
                filtered.append(product)
        return filtered
    
    def _filter_by_brand(self, products: List[Dict], brand: str) -> List[Dict]:
        """Lọc sản phẩm theo brand"""
        brand_lower = brand.lower()
        filtered = []
        for product in products:
            product_name = product.get('name', '').lower()
            product_desc = product.get('description', '').lower()
            
            if brand_lower in product_name or brand_lower in product_desc:
                filtered.append(product)
        return filtered
    
    def _filter_by_price(self, products: List[Dict], price_range: str) -> List[Dict]:
        """Lọc sản phẩm theo khoảng giá"""
        price_mapping = {
            "low": (0, 50000),
            "medium": (50000, 200000),
            "high": (200000, float('inf'))
        }
        
        if price_range in price_mapping:
            min_price, max_price = price_mapping[price_range]
            filtered = []
            for product in products:
                price = product.get('price', 0)
                if min_price <= price <= max_price:
                    filtered.append(product)
            return filtered
        return products

class EnhancedRecommender:
    def __init__(self, products):
        self.products = products
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500)
        self.product_filter = SmartProductFilter(products)
        self.product_features = None
        self._prepare_product_data()
        
    def _prepare_product_data(self):
        """Chuẩn bị dữ liệu sản phẩm"""
        self.product_texts = []
        for product in self.products:
            text_data = f"{product.get('name', '')} {product.get('category', '')} {product.get('description', '')} {product.get('content', '')}"
            self.product_texts.append(text_data)
    
    def extract_features(self):
        """Trích xuất đặc trưng"""
        if not self.product_texts:
            self._prepare_product_data()
            
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.product_texts)
            n_components = min(50, len(self.products)-1)
            
            if n_components > 0:
                self.svd = TruncatedSVD(n_components=n_components)
                self.product_features = self.svd.fit_transform(tfidf_matrix)
            else:
                self.product_features = tfidf_matrix.toarray()
                
        except Exception as e:
            print(f"❌ Lỗi extract features: {e}")
            # Fallback: random features
            self.product_features = np.random.rand(len(self.products), 50)
    
    def recommend(self, analysis_result: Dict, top_k: int = 10) -> List[Dict]:
        """Gợi ý sản phẩm dựa trên phân tích chi tiết"""
        if self.product_features is None:
            self.extract_features()
            
        # Bước 1: Tìm kiếm cơ bản dựa trên processed_query
        query = analysis_result["processed_query"]
        if query and query.strip() != "":
            try:
                query_vec = self.tfidf_vectorizer.transform([query])
                
                if hasattr(self, 'svd'):
                    query_vec_reduced = self.svd.transform(query_vec)
                    similarities = cosine_similarity(query_vec_reduced, self.product_features)[0]
                else:
                    similarities = cosine_similarity(query_vec, self.product_features)[0]
                
                # Kết hợp sản phẩm và similarity scores
                scored_products = []
                for idx, similarity in enumerate(similarities):
                    if similarity > 0.01:  # Ngưỡng tối thiểu
                        product = self.products[idx].copy()
                        product['similarity_score'] = float(similarity)
                        scored_products.append(product)
                
                # Sắp xếp theo similarity
                scored_products.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
                
            except Exception as e:
                print(f"❌ Lỗi tìm kiếm: {e}")
                scored_products = self.products.copy()
        else:
            scored_products = self.products.copy()
        
        # Bước 2: Áp dụng các bộ lọc thông minh
        filtered_products = self.product_filter.apply_filters(scored_products, analysis_result)
        
        # Bước 3: Ưu tiên theo intent
        final_results = self._prioritize_by_intent(filtered_products, analysis_result["intent"])
        
        return final_results[:top_k]
    
    def _prioritize_by_intent(self, products: List[Dict], intent: str) -> List[Dict]:
        """Ưu tiên sản phẩm dựa trên ý định"""
        if intent == "budget":
            # Ưu tiên giá rẻ
            return sorted(products, key=lambda x: x.get('price', float('inf')))
        elif intent == "compare":
            # Giữ nguyên thứ tự similarity
            return products
        else:
            # Mặc định: similarity score
            return sorted(products, key=lambda x: x.get('similarity_score', 0), reverse=True)

class SupermarketAI:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_loader = DataLoader(data_path)  # ĐÃ ĐƯỢC ĐỊNH NGHĨA
        self.gpt_analyzer = GPTQueryAnalyzer()
        self.recommender = None
        
    def initialize_system(self):
        """Khởi tạo hệ thống"""
        print("🔄 Đang khởi tạo hệ thống AI thông minh...")
        
        # Load dữ liệu
        products, _ = self.data_loader.load_data()
        
        if not products:
            print("❌ Không có dữ liệu sản phẩm")
            return False
            
        # Khởi tạo recommender
        self.recommender = EnhancedRecommender(products)
        
        # Train model
        print("🔄 Training AI model...")
        self.recommender.extract_features()
        
        print("✅ Hệ thống AI thông minh đã sẵn sàng!")
        return True
        
    def process_query(self, user_query: str, top_k: int = 5) -> Dict:
        """Xử lý query người dùng với GPT"""
        if not self.recommender:
            print("❌ Hệ thống chưa được khởi tạo")
            return None
        
        # Phân tích query với GPT
        analysis = self.gpt_analyzer.analyze_query(user_query)
        
        print(f"\n🔍 PHÂN TÍCH GPT:")
        print(f"   Query gốc: '{analysis['original_query']}'")
        print(f"   Query xử lý: '{analysis['processed_query']}'")
        print(f"   Ý định: {analysis['intent']}")
        print(f"   Danh mục: {analysis['category']}")
        print(f"   Thương hiệu: {analysis['brand']}")
        print(f"   Thuộc tính: {analysis['attributes']}")
        
        # Lấy recommendations
        recommendations = self.recommender.recommend(analysis, top_k)
        
        return {
            'analysis': analysis,
            'recommendations': recommendations,
            'total_results': len(recommendations)
        }
    
    def display_recommendations(self, result):
        """Hiển thị kết quả recommendations"""
        if not result or not result['recommendations']:
            print("❌ Không tìm thấy sản phẩm phù hợp")
            return
            
        analysis = result['analysis']
        
        print(f"\n🎯 KẾT QUẢ CHO: '{analysis['original_query']}'")
        print(f"📊 Tìm thấy {result['total_results']} sản phẩm phù hợp")
        print("=" * 80)
        
        for i, product in enumerate(result['recommendations'], 1):
            print(f"{i}. {product['name']}")
            print(f"   📂 Danh mục: {product.get('category', 'N/A')}")
            print(f"   💰 Giá: {product.get('price', 'N/A'):,} VND")
            print(f"   📦 Tồn kho: {product.get('quantity', 'N/A')}")
            
            # Hiển thị các scores nếu có
            if 'similarity_score' in product:
                print(f"   🔍 Độ phù hợp: {product['similarity_score']:.3f}")
                
            print("-" * 50)

# Hàm main để test
def main():
    data_path = "data.json"  # Thay đổi đường dẫn đến file data của bạn
    
    # Khởi tạo hệ thống
    ai_system = SupermarketAI(data_path)
    
    if not ai_system.initialize_system():
        return
    
    # Test các query đa dạng
    test_queries = [
        "Tôi muốn mua bơ lạt Anchor giá rẻ",
    ]
    
    print("\n" + "="*80)
    print("🧪 TEST HỆ THỐNG AI THÔNG MINH VỚI GPT")
    print("="*80)
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"🧪 TEST: '{query}'")
        print('='*50)
        
        result = ai_system.process_query(query, top_k=3)
        ai_system.display_recommendations(result)

if __name__ == "__main__":
    main()