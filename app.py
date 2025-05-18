import os
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from pymongo import MongoClient
from bson import ObjectId
from config import Config
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename # For secure file uploads
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import datetime
from datetime import timezone # Added for timezone-aware datetime
# import pytesseract # No longer needed
# from PIL import Image, ImageEnhance # No longer needed for direct Tesseract processing here
import re 

# --- New OCR Imports ---
import cv2
import numpy as np
from ocr_extractor import Extractor # Assuming ocr_extractor.py is in the same directory or accessible

# --- Tesseract OCR Configuration --- (This section can be removed or commented out)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = 'uploads/cccd_images'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- OCR Extractor Instance (Initialize once) ---
# It's better to initialize heavy models once if possible.
# If your server runs in multiple processes, this might re-initialize per process.
# For true single initialization in a multi-process/threaded server, other strategies might be needed.
ocr_pipeline = None

def get_ocr_pipeline():
    global ocr_pipeline
    if ocr_pipeline is None:
        print("Initializing OCR Extractor Pipeline...")
        try:
            ocr_pipeline = Extractor()
            print("OCR Extractor Pipeline initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize OCR Extractor Pipeline: {e}")
            # To prevent re-trying initialization on every call if it fails badly:
            # ocr_pipeline = "FAILED_TO_INITIALIZE" # Sentinel value
    # elif ocr_pipeline == "FAILED_TO_INITIALIZE":
    #     return None
    return ocr_pipeline

# Initialize it once at startup (for single-process dev servers primarily)
# For production, consider lazy loading as above or specific server hooks if available.
# get_ocr_pipeline() # You might call this here or rely on lazy loading in process_cccd_image_ocr


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = "Vui lòng đăng nhập để truy cập trang này."
login_manager.login_message_category = "info"

@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now(timezone.utc)}

client = MongoClient(app.config['MONGO_URI'])
db = client.get_default_database()
if db is None:
    db_name_from_uri = app.config['MONGO_URI'].split('/')[-1].split('?')[0]
    db = client[db_name_from_uri if db_name_from_uri else "quick_room_db"]
items_collection = db["items"]
users_collection = db["users"]

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data["_id"])
        self.cccd = user_data.get("cccd")
        self.email = user_data.get("email")
        self.full_name = user_data.get("full_name")
        self.date_of_birth = user_data.get("date_of_birth")
        self.sex = user_data.get("sex")
        self.address = user_data.get("address")
        self.place_of_origin = user_data.get("place_of_origin")
        self.cccd_image_path = user_data.get("cccd_image_path")

    @staticmethod
    def get(user_id):
        user_data = users_collection.find_one({"_id": ObjectId(user_id)})
        return User(user_data) if user_data else None

    @staticmethod
    def find_by_cccd(cccd):
        return users_collection.find_one({"cccd": cccd})

    @staticmethod
    def find_by_email(email):
        return users_collection.find_one({"email": email})

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

# --- New OCR Processing Function ---
def process_cccd_image_ocr(image_path):
    """
    Processes a CCCD image using the new Extractor (PaddleOCR + VietOCR).
    """
    extractor_instance = get_ocr_pipeline()
    if not extractor_instance or extractor_instance == "FAILED_TO_INITIALIZE":
        print("OCR pipeline not available or failed to initialize.")
        return None

    try:
        print(f"[INFO] Starting OCR process for: {image_path}")
        # The get_information method in Extractor now handles reading the image
        extracted_data = extractor_instance.get_information(image_path)
        print(f"[INFO] OCR process completed. Extracted: {extracted_data}")
        
        # Map keys from Extractor to keys expected by the rest of the app if they differ
        # Current Extractor output keys: cccd, full_name, date_of_birth, sex, place_of_origin, address, nationality
        # App user User class keys: cccd, full_name, date_of_birth, sex, address, place_of_origin
        # No direct mapping needed if keys are the same or subset is fine.
        
        if not extracted_data or (not extracted_data.get('cccd') and not extracted_data.get('full_name')):
            print("[INFO] OCR extraction did not yield significant data (CCCD or Name missing).")
            return None
            
        return extracted_data

    except Exception as e:
        print(f"Lỗi trong quá trình OCR (Extractor): {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

@app.route('/')
def route_index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    form_data_for_template = {}

    if request.method == 'POST':
        cccd = request.form.get('cccd')
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        sex = request.form.get('sex')
        place_of_origin = request.form.get('place_of_origin')
        address = request.form.get('address') 
        date_of_birth = request.form.get('date_of_birth') 
        cccd_image_file = request.files.get('cccd_image')

        form_data_for_template = request.form.to_dict()
        cccd_image_filename = None

        if cccd_image_file and cccd_image_file.filename != '':
            if allowed_file(cccd_image_file.filename):
                unique_prefix = f"{cccd or 'temp'}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                filename = secure_filename(f"{unique_prefix}_{cccd_image_file.filename}")
                cccd_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    cccd_image_file.save(cccd_image_path)
                    cccd_image_filename = filename 
                    
                    extracted_ocr_data = process_cccd_image_ocr(cccd_image_path)
                    if extracted_ocr_data:
                        flash("Đã nhận diện thông tin từ ảnh CCCD. Vui lòng kiểm tra và xác nhận lại các trường.", "info")
                        for key, value in extracted_ocr_data.items():
                            # Use .get(key) for form_data_for_template as well for safety
                            if value and (not form_data_for_template.get(key) or form_data_for_template.get(key) == 'temp'):
                                form_data_for_template[key] = value
                        if extracted_ocr_data.get('cccd') and not cccd:
                            cccd = extracted_ocr_data.get('cccd')      
                    else:
                        flash("OCR không nhận diện được nhiều thông tin từ ảnh CCCD. Vui lòng điền thủ công.", "warning")
                except Exception as e:
                    flash(f"Lỗi lưu hoặc xử lý ảnh CCCD: {str(e)}", "danger")
            else:
                flash("Định dạng file ảnh CCCD không hợp lệ. Chỉ chấp nhận .png, .jpg, .jpeg, .gif", "warning")
        
        cccd = form_data_for_template.get('cccd', cccd)
        full_name = form_data_for_template.get('full_name', full_name)
        email = form_data_for_template.get('email', email)
        date_of_birth = form_data_for_template.get('date_of_birth', date_of_birth)
        sex = form_data_for_template.get('sex', sex)
        place_of_origin = form_data_for_template.get('place_of_origin', place_of_origin)
        address = form_data_for_template.get('address', address)

        if not all([cccd, full_name, email, date_of_birth, sex, address, place_of_origin, password, confirm_password]):
            flash("Vui lòng điền đầy đủ các trường bắt buộc được đánh dấu *.", "danger")
            return render_template('register.html', form_data=form_data_for_template) 
        
        if len(cccd) not in [9, 12] or not cccd.isdigit():
            flash("Số CCCD không hợp lệ (cần 9 hoặc 12 số).", "danger")
            return render_template('register.html', form_data=form_data_for_template)
        
        if password != confirm_password:
            flash("Mật khẩu và xác nhận mật khẩu không khớp.", "danger")
            return render_template('register.html', form_data=form_data_for_template)

        if len(password) < 6:
            flash("Mật khẩu phải có ít nhất 6 ký tự.", "danger")
            return render_template('register.html', form_data=form_data_for_template)
        
        try:
            if date_of_birth: # Ensure date_of_birth is not None or empty
                datetime.datetime.strptime(date_of_birth, '%d/%m/%Y')
            else:
                flash("Ngày sinh không được để trống.", "danger")
                return render_template('register.html', form_data=form_data_for_template)
        except ValueError:
            flash("Định dạng ngày sinh không hợp lệ. Vui lòng dùng DD/MM/YYYY.", "danger")
            return render_template('register.html', form_data=form_data_for_template)

        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            flash("Địa chỉ email không hợp lệ.", "danger")
            return render_template('register.html', form_data=form_data_for_template)

        existing_user_by_cccd = User.find_by_cccd(cccd)
        if existing_user_by_cccd:
            flash("Số CCCD này đã được đăng ký.", "warning")
            return render_template('register.html', form_data=form_data_for_template)
        
        existing_user_by_email = User.find_by_email(email)
        if existing_user_by_email:
            flash("Địa chỉ email này đã được đăng ký.", "warning")
            return render_template('register.html', form_data=form_data_for_template)
        
        hashed_password = generate_password_hash(password)
        user_document = {
            "cccd": cccd, "full_name": full_name, "email": email, "date_of_birth": date_of_birth,
            "sex": sex, "place_of_origin": place_of_origin, "address": address,
            "password": hashed_password,
            "cccd_image_path": cccd_image_filename
        }
        users_collection.insert_one(user_document)
        flash("Đăng ký thành công! Vui lòng đăng nhập.", "success")
        return redirect(url_for('login'))
            
    return render_template('register.html', form_data=form_data_for_template)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form_data_for_template = {}
    if request.method == 'POST':
        cccd_typed = request.form.get('cccd')
        password = request.form.get('password')
        cccd_image_file = request.files.get('cccd_image') 
        form_data_for_template = {'cccd': cccd_typed}
        final_cccd_to_check = cccd_typed
        temp_image_path_for_login = None

        if cccd_image_file and cccd_image_file.filename != '':
            if allowed_file(cccd_image_file.filename):
                filename = secure_filename(f"login_{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}_{cccd_image_file.filename}")
                temp_image_path_for_login = os.path.join(app.config['UPLOAD_FOLDER'], filename) 
                try:
                    cccd_image_file.save(temp_image_path_for_login)
                    extracted_ocr_data = process_cccd_image_ocr(temp_image_path_for_login)
                    if extracted_ocr_data and extracted_ocr_data.get('cccd'):
                       final_cccd_to_check = extracted_ocr_data.get('cccd')
                       form_data_for_template['cccd'] = final_cccd_to_check
                       flash("Đã nhận diện CCCD từ ảnh. Vui lòng nhập mật khẩu.", "info")
                    elif extracted_ocr_data :
                        flash("OCR không nhận diện được số CCCD từ ảnh. Sử dụng CCCD đã nhập (nếu có).", "warning")
                    else:
                        flash("Không thể xử lý ảnh CCCD bằng OCR, sử dụng CCCD đã nhập (nếu có).", "warning")
                except Exception as e:
                    flash(f"Lỗi xử lý ảnh CCCD: {str(e)}", "danger")
                finally:
                    if temp_image_path_for_login and os.path.exists(temp_image_path_for_login): 
                        try: os.remove(temp_image_path_for_login)
                        except: pass # Silently pass remove error for temp file
            else:
                 flash("Định dạng file ảnh CCCD không hợp lệ khi đăng nhập.", "warning")
        
        if not final_cccd_to_check:
             flash("Vui lòng nhập CCCD hoặc tải ảnh CCCD có thể nhận diện.", "danger")
             return render_template('login.html', form_data=form_data_for_template)
        if not password:
            flash("Vui lòng nhập mật khẩu.", "danger")
            return render_template('login.html', form_data=form_data_for_template)

        user_data_from_db = User.find_by_cccd(final_cccd_to_check)
        if user_data_from_db and check_password_hash(user_data_from_db.get("password", ""), password):
            user_obj = User(user_data_from_db) 
            login_user(user_obj)
            flash("Đăng nhập thành công!", "success")
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash("Số CCCD hoặc mật khẩu không chính xác.", "danger")
            return render_template('login.html', form_data=form_data_for_template)
            
    return render_template('login.html', form_data=form_data_for_template)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Bạn đã đăng xuất.", "info")
    return redirect(url_for('login'))

@app.route('/home')
@login_required
def home():
    return render_template('home.html')

@app.route('/api/ocr-cccd', methods=['POST'])
def ocr_cccd_api():
    if 'cccd_image' not in request.files:
        return jsonify({"error": "Không có file ảnh nào được gửi lên."}), 400
    file = request.files['cccd_image']
    if file.filename == '':
        return jsonify({"error": "Không có file nào được chọn."}), 400
    if file and allowed_file(file.filename):
        user_identifier = current_user.id if current_user.is_authenticated else 'anon'
        temp_filename_for_ocr = f"ocr_temp_{user_identifier}_{secure_filename(file.filename)}"
        temp_image_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename_for_ocr)
        try:
            file.save(temp_image_path)
            extracted_data = process_cccd_image_ocr(temp_image_path)
            if extracted_data:
                return jsonify({"success": True, "data": extracted_data}), 200
            else:
                return jsonify({"success": False, "message": "Không nhận diện được thông tin từ ảnh."}), 200
        except Exception as e:
            print(f"Lỗi API OCR: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return jsonify({"error": f"Lỗi xử lý ảnh: {str(e)}"}), 500
        finally:
            if os.path.exists(temp_image_path):
                try: os.remove(temp_image_path)
                except Exception as e_remove: print(f"Lỗi xoá file tạm OCR: {e_remove}")
    else:
        return jsonify({"error": "Định dạng file không hợp lệ."}), 400

if __name__ == '__main__':
    get_ocr_pipeline() # Pre-initialize OCR pipeline when running directly
    app.run(debug=app.config['FLASK_DEBUG'], host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))