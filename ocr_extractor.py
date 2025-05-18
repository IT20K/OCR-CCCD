import os
import re
import json
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import time # Added for __main__ block

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

class Extractor:
    def __init__(self):
        # VietOCR configuration
        self.config = Cfg.load_config_from_name('vgg_seq2seq')
        # Ensure the path to weights is correct. Assuming seq2seqocr.pth is in the same directory.
        self.config['weights'] = os.path.join(CURRENT_DIR, "seq2seqocr.pth") 
        self.config['cnn']['pretrained'] = False
        self.config['device'] = 'cpu' # Change to 'cuda:0' if using GPU

        # Initialize PaddleOCR (for text detection)
        # lang='en' is often used for detection as it's robust for layout, 
        # recognition will be handled by VietOCR. Or use 'vi' if it yields better detection boxes.
        self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='vi', show_log=False) 
        
        # Initialize VietOCR (for text recognition)
        self.viet_ocr_detector = Predictor(self.config)

    def detection(self, frame_path_or_img):
        """
        Detects text regions in the image.
        Args:
            frame_path_or_img: Path to the image or a NumPy array (BGR format from cv2.imread).
        Returns:
            A list of detected bounding boxes and recognized text from PaddleOCR.
            Each item is [[box_points], [recognized_text, confidence]].
            We primarily use the box_points for VietOCR.
        """
        result = self.paddle_ocr.ocr(frame_path_or_img, cls=True)
        return result[0] if result and result[0] is not None else []


    def warp_and_recognize_text(self, frame_bgr, box_points):
        """
        Warps the detected text region and recognizes text using VietOCR.
        Args:
            frame_bgr: The full image frame (NumPy array in BGR format).
            box_points: List of 4 points defining the bounding box of the text region.
                        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
        Returns:
            A tuple containing (recognized_string, bounding_box_coordinates).
            Returns (None, box_points) if warping or recognition fails.
        """
        try:
            # Ensure points are in float32 for getPerspectiveTransform
            rect = np.array(box_points, dtype="float32")
            
            # Order the points: top-left, top-right, bottom-right, bottom-left
            # PaddleOCR returns points in this order: top-left, top-right, bottom-right, bottom-left
            tl, tr, br, bl = rect

            # Compute the width of the new image
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            # Compute the height of the new image
            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            if maxWidth == 0 or maxHeight == 0:
                # print(f"Warning: Invalid dimensions for warping: {maxWidth}x{maxHeight}. Box: {box_points}")
                return None, box_points


            # Define the destination points for the warped image
            dst = np.array([
                [0, 0],
                [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1],
                [0, maxHeight - 1]], dtype="float32")

            # Compute the perspective transform matrix
            M = cv2.getPerspectiveTransform(rect, dst)
            # Apply the perspective warp
            warped_bgr = cv2.warpPerspective(frame_bgr, M, (maxWidth, maxHeight))

            # Convert BGR (OpenCV) to RGB (PIL) for VietOCR
            warped_rgb = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(warped_rgb)
            
            # Recognize text using VietOCR
            recognized_text = self.viet_ocr_detector.predict(pil_image)
            return recognized_text, box_points
        except Exception as e:
            # print(f"Error during warp_and_recognize_text: {e}. Box: {box_points}")
            return None, box_points

    def get_information(self, image_path):
        """
        Main function to extract all information from a CCCD image.
        """
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Error: Could not read image from {image_path}")
                return {}
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return {}

        detected_lines = self.detection(image_path) # Pass image_path or frame
        if not detected_lines:
            print("No text detected by PaddleOCR.")
            return {}

        recognized_results = []
        for line_info in detected_lines:
            # line_info is [[box_points], [recognized_text_paddle, confidence]]
            box = line_info[0] # box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # We use VietOCR for better recognition
            vietocr_text, _ = self.warp_and_recognize_text(frame, box)
            if vietocr_text:
                recognized_results.append({'text': vietocr_text, 'box': box})
        
        # Sort results by vertical position (top-to-bottom), then horizontal (left-to-right)
        # This helps in associating labels with values that appear below or to the right.
        recognized_results.sort(key=lambda r: (r['box'][0][1], r['box'][0][0]))


        # print("----------- Recognized Texts (VietOCR) -----------")
        # for res in recognized_results:
        #     print(f"Box: {res['box']}\\tText: {res['text']}")
        # print("-------------------------------------------------")

        result_data = {
            'cccd': None,
            'full_name': None,
            'date_of_birth': None,
            'sex': None,
            'place_of_origin': None,
            'address': None,
            'nationality': None # Added nationality as it's often on CCCD
        }

        # Regex patterns (more specific)
        # regex_cccd = r'(?:Số/No|Số|No|CCCD|sá/Ao)[.:\s]*([0-9]{12})' # 12 digits for CCCD
        regex_cccd = r'([0-9]{12})' # More general for 12 digits if label is missed
        regex_dob = r'(\d{1,2}[/.-]\d{1,2}[/.-]\d{4})' # DD/MM/YYYY or similar

        temp_texts_for_extraction = [res['text'] for res in recognized_results]
        full_ocr_text = "\\n".join(temp_texts_for_extraction) # For easier multi-line regex if needed

        # --- Field Extraction Logic ---

        # 1. CCCD (ID Number)
        for res in recognized_results:
            text = res['text']
            # Try to find label + number first
            match = re.search(r'(?:Số|No|CCCD|sá/Ao|Citizen ldentity Card|ldentity Card No)[.:\s]*([0-9]{12})', text, re.IGNORECASE)
            if match:
                result_data['cccd'] = match.group(1)
                break
        if not result_data['cccd']: # If not found with label, try to find any 12-digit number
            for res in recognized_results:
                text = res['text']
                match = re.search(regex_cccd, text)
                if match:
                    result_data['cccd'] = match.group(1)
                    break
        
        # 2. Full Name
        # Look for "Họ và tên" or "Full name", then expect name on the next few lines.
        for i, res in enumerate(recognized_results):
            text = res['text'].lower()
            if "họ và tên" in text or "full name" in text:
                # Check next 1-2 lines for a capitalized name
                for j in range(1, 3): # Look at next 2 lines
                    if i + j < len(recognized_results):
                        next_text = recognized_results[i+j]['text']
                        # A name is usually multiple capitalized words, no digits, not a label for another field.
                        if (len(next_text.split()) >= 2 and 
                            next_text.isupper() and # Common for names on ID
                            not any(char.isdigit() for char in next_text) and
                            len(next_text) > 5 and # Avoid short random uppercase texts
                            not any(stop_word in next_text.lower() for stop_word in ["ngày sinh", "giới tính", "quê quán", "nơi thường"])):
                            result_data['full_name'] = next_text
                            break
                if result_data['full_name']:
                    break
        
        # 3. Date of Birth
        for res in recognized_results:
            text = res['text']
            # Search for label and DOB pattern
            match_label_dob = re.search(r'(?:Ngày sinh|Date of birth|Date of bíth|DOB)[.:\s]*' + regex_dob, text, re.IGNORECASE)
            if match_label_dob:
                result_data['date_of_birth'] = match_label_dob.group(1)
                break
            # If no label, just search for DOB pattern directly in lines that might contain it
            match_dob_only = re.search(regex_dob, text)
            if match_dob_only and not result_data['date_of_birth']: # Only if not already found
                 # Basic check to avoid grabbing expiry dates by mistake if it's a standalone date
                if not ("có giá trị đến" in text.lower() or "expiry" in text.lower() or "date of issue" in text.lower()):
                    result_data['date_of_birth'] = match_dob_only.group(1)
                    # Don't break here, a labeled one is preferred.

        # 4. Sex (Gender)
        for res in recognized_results:
            text = res['text']
            match = re.search(r'(?:Giới tính|Sex)[.:\s]*(Nam|Nữ|Male|Female)', text, re.IGNORECASE)
            if match:
                sex_val = match.group(1).lower()
                result_data['sex'] = 'Nam' if sex_val in ['nam', 'male'] else 'Nữ' if sex_val in ['nữ', 'female'] else None
                break
        
        # 5. Nationality
        for res in recognized_results:
            text = res['text']
            match = re.search(r'(?:Quốc tịch|Nationality)[.:\s]*(.+)', text, re.IGNORECASE)
            if match:
                nationality_text = match.group(1).strip()
                if "việt nam" in nationality_text.lower():
                     result_data['nationality'] = "Việt Nam"
                else:
                     result_data['nationality'] = nationality_text.title()
                break
        if not result_data['nationality']: # Fallback if label is missed but "Việt Nam" appears clearly
            for res in recognized_results:
                if "Việt Nam" in res['text'] and len(res['text']) < 20: # Check if it's a standalone "Việt Nam"
                    result_data['nationality'] = "Việt Nam"
                    break
        
        # 6. Place of Origin (Quê quán) & 7. Place of Residence (Nơi thường trú)
        # These are often multi-line and require looking at subsequent lines after a label.
        # This is a simplified approach; more robust would be to analyze spatial relationships of boxes.
        
        def extract_multiline_field(start_keywords, end_keywords, all_results, current_index):
            extracted_lines = []
            # The line containing the keyword is current_index
            # Start collecting from current_index + 1 or the same line if value is on it
            
            # Check if value is on the same line as the keyword
            text_on_keyword_line = all_results[current_index]['text']
            keyword_line_lower = text_on_keyword_line.lower()

            for kw in start_keywords:
                kw_lower = kw.lower()
                if kw_lower in keyword_line_lower:
                    idx_kw = keyword_line_lower.find(kw_lower)
                    if idx_kw != -1:
                        # Extract text after the keyword, then strip leading chars
                        # More robust stripping
                        first_part_candidate = text_on_keyword_line[idx_kw + len(kw):]
                        first_part = re.sub(r"^[\\s.:/-]+", "", first_part_candidate)
                        if first_part:
                             extracted_lines.append(first_part)
                    break # Found a start keyword on this line

            # Determine how many more lines to collect
            # If something was found on the keyword line, collect up to 2 more lines.
            # If keyword line had no value, collect up to 3 lines.
            lines_to_collect_further = 2
            if not extracted_lines: # True if first_part was empty or no keyword match on current line for value
                lines_to_collect_further = 3

            actual_lines_collected_further = 0
            for k in range(current_index + 1, len(all_results)):
                next_text_candidate = all_results[k]['text']
                
                # Check for stop conditions before processing the line
                # Stop if an end keyword (specific to this field type) or any other major field label starts
                stop_due_to_end_keyword = any(end_kw.lower() in next_text_candidate.lower() for end_kw in end_keywords)
                # Using common_end_keywords which is defined outside this function but accessible via closure or by passing if necessary
                # For now, assuming common_end_keywords are the main stopping labels for ANY multiline field, 
                # and end_keywords are specific to differentiate between origin/address themselves.
                stop_due_to_major_label = any(label_kw.lower() in next_text_candidate.lower() for label_kw in 
                                            ["họ và tên", "ngày sinh", "giới tính", "quốc tịch", "số cccd", "đặc điểm", "có giá trị đến"])

                if stop_due_to_end_keyword or stop_due_to_major_label:
                    break

                next_text = next_text_candidate.strip()
                if next_text and len(next_text) > 2: # Non-empty and not too short
                    extracted_lines.append(next_text)
                    actual_lines_collected_further += 1
                
                if actual_lines_collected_further >= lines_to_collect_further:
                    break
            
            return " ".join(extracted_lines).strip() if extracted_lines else None

        common_end_keywords = ["họ và tên", "full name", "ngày sinh", "date of birth", "giới tính", "sex", "quốc tịch", "nationality", "số cccd", "citizen identity card", "đặc điểm", "personal identification", "có giá trị đến", "date of expiry"]

        for i, res in enumerate(recognized_results):
            text_lower = res['text'].lower()
            # Added more keyword variations
            origin_keywords = ["quê quán", "place of origin", "p!ace of origin", "quê quän", "quê quán/", "auê quán", "quê quán :"]
            address_keywords = ["nơi thường trú", "place of residence", "fplace of residence", "nơi thưòng trú", "nơi thuong trú", "nơi thường trú/", "noi thường trú", "nơi thường trú :", "địa chỉ:"]

            if not result_data['place_of_origin'] and any(kw in text_lower for kw in origin_keywords):
                origin_end_kws = common_end_keywords + [kw for kw in address_keywords] # Stop if address starts
                val = extract_multiline_field(origin_keywords, origin_end_kws, recognized_results, i)
                if val: result_data['place_of_origin'] = val

            if not result_data['address'] and any(kw in text_lower for kw in address_keywords):
                address_end_kws = common_end_keywords + ([kw for kw in origin_keywords] if not result_data['place_of_origin'] else []) # Stop if origin starts (if not found yet)
                val = extract_multiline_field(address_keywords, address_end_kws, recognized_results, i)
                if val: result_data['address'] = val
        
        # Final cleanup for extracted values (remove residual labels, leading/trailing chars)
        for key, value in result_data.items():
            if isinstance(value, str):
                cleaned_value = value.strip()
                # Remove common prefixes that might have been captured if regex was too greedy
                prefixes_to_remove = [
                    "Số/No.:", "Số/No", "Số:", "No.:", "sá/Ao:", "So:", "S6:",
                    "Họ và tên / Full name:", "Họ và tên:", "Full name:", "Họ tên:", "Ho và tên:",
                    "Ngày sinh / Date of birth:", "Ngày sinh:", "Date of birth:", "Date of bíth:", "Ngày,:sinh / Date of birth", "Ngay sinh:",
                    "Giới tính / Sex:", "Giới tính:", "Sex:",
                    "Quốc tịch / Nationality:", "Quốc tịch:", "Nationality:", "Quoc tich:",
                    "Quê quán / Place of origin:", "Quê quán:", "Place of origin:", "P!ace of origin:", "Que quan / Place of origin.", "Quê quán :",
                    "Nơi thường trú / Place of residence:", "Nơi thường trú:", "Place of residence:", "fPlace of residence:", "I Place of residence:", "Nơi thường trú :", "Địa chỉ:",
                    "Đặc điểm nhận dạng/Personal identification:", "Đặc điểm nhận dạng:", "Personal identification:",
                    "Ngày, tháng, năm/Date, month, year:",
                    "|" # Common OCR artifact
                ]
                for prefix in prefixes_to_remove:
                    if cleaned_value.lower().startswith(prefix.lower()):
                        cleaned_value = cleaned_value[len(prefix):]
                        # More aggressive stripping after prefix removal
                        cleaned_value = re.sub(r"^[\\s.:/-]+", "", cleaned_value)
                
                # General trailing punctuation/artifact removal (e.g., from 'Nationality;')
                cleaned_value = re.sub(r"[;.,:]$", "", cleaned_value.strip()).strip()

                result_data[key] = cleaned_value if cleaned_value else None


        # print(f"Extracted Information: {result_data}")
        return result_data

# Example usage (for testing ocr_extractor.py directly)
if __name__ == '__main__':
    # This is a placeholder for an actual image path for testing
    test_image_path = 'path_to_your_test_cccd_image.jpg' 
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}. Please provide a valid path.")
    else:
        print(f"Initializing Extractor...")
        idcard_extractor = Extractor()
        print(f"Extractor initialized. Processing image: {test_image_path}")
        
        start_time = time.time()
        info = idcard_extractor.get_information(test_image_path)
        end_time = time.time()
        
        print(f"\\n--- Extracted Information (from {test_image_path}) ---")
        print(json.dumps(info, indent=4, ensure_ascii=False))
        print(f"Processing time: {end_time - start_time:.2f} seconds") 