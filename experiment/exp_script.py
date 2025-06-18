import ollama
import time
import re
import csv
import json
from datetime import datetime, timedelta, timezone

# 設定模型名稱
model = "llama2:13b-chat"

def get_taiwan_time():
    """取得台灣時間 (GMT+8)"""
    tw_timezone = timezone(timedelta(hours=8))
    tw_time = datetime.now(tw_timezone)
    return tw_time.strftime("%Y-%m-%d %H:%M:%S")



user_input = ""

data_text = """Course topic of Week 1 [2025/02/17–02/21]:Course introduction, basic image processing concepts, reading and displaying grayscale images
Course topic of Week 2 [2025/02/24–02/28]:Basic operations in NumPy and OpenCV (matrix operations, image I/O, drawing)
Course topic of Week 3 [2025/03/03–03/07]:Convolution operations and edge detection (Sobel, Prewitt)
Course topic of Week 4 [2025/03/10–03/14]:Smoothing and blurring filters (mean filter, Gaussian filter)
Course topic of Week 5 [2025/03/17–03/21]:Basics of Convolutional Neural Networks (CNN) and their application in image classification
Course topic of Week 6 [2025/03/24–03/28]:Binarization and thresholding (including Otsu's method)
Course topic of Week 7 [2025/03/31–04/04]:Local operations: morphological operations (dilation, erosion, opening/closing)
Course topic of Week 8 [2025/04/07–04/11]:Geometric transformations (translation, rotation, scaling, affine, perspective)
Course topic of Week 9 [2025/04/14–04/18]:Midterm exam (held on 2025/04/14, covering Weeks 1–8)
Course topic of Week 10	[2025/04/21–04/25]:Image contour detection and analysis (cv2.findContours, area/perimeter/approximation)
Course topic of Week 11	[2025/04/28–05/02]:Connected components analysis
Course topic of Week 12	[2025/05/05–05/09]:Edge detection and Hough Transform (Hough lines/circles)
Course topic of Week 13	[2025/05/12–05/16]:Color space conversion and image segmentation (BGR↔HSV, k-means segmentation)
Course topic of Week 14	[2025/05/19–05/23]:Feature detection and description (Harris, SIFT, ORB concepts)
Course topic of Week 15	[2025/05/26–05/30]:Project mentoring week (group discussions and progress checks)
Course topic of Week 16	[2025/06/02–06/06]:Image classification and shape recognition (basic CNN / logical shape classification)
Course topic of Week 17	[2025/06/09–06/13]:Final project presentations and summary preparation
Course topic of Week 18	[2025/06/16–06/20]:Final exam (held on 2025/06/16, covering Weeks 10–16)

Announcement 1 [2025/03/10]:Class attendance will affect your final grade. Do not skip class or sign in for others without valid reasons.
Announcement 2 [2025/03/12]:This week's class will cover the basic structure and applications of Convolutional Neural Networks (CNN). Please bring your laptop.
Announcement 3 [2025/03/14]:Homework 1 has been released. Please download it from the course platform. The deadline is 4/15 (Monday) at 23:59.
Announcement 4 [2025/03/18]:You must write your own code for the assignments. Using pre-trained models or solutions is prohibited.
Announcement 5 [2025/03/20]:Please follow the naming format for submissions, e.g., studentID_hw1.py.
Announcement 6 [2025/03/25]:Course videos have been uploaded to the teaching platform. Please watch them and complete the video quiz this week.
Announcement 7 [2025/04/01]:FAQs for Homework 1 are available. Check the announcements section to avoid repeated questions.
Announcement 8 [2025/04/05]:A TA session will be held this Friday afternoon. Students with questions are welcome to attend.
Announcement 9 [2025/04/08]:The **midterm exam** will be held next Monday (2025/04/14), covering the first eight weeks. Please prepare well.
Announcement 10 [2025/05/01]:Please submit your final project topic by May 10, 2025. Refer to the attachment in the announcement for the required format.
Announcement 11 [2025/06/08]:This week’s topic is “Edge Detection and Hough Transform.” Please preview the relevant math concepts.
Announcement 12 [2025/06/09]:If you need to submit late assignments, email the instructor and TA in advance to explain the reason.
Announcement 13 [2025/06/10]:Final project presentation order has been announced. Please submit your project proposal by next Friday.
Announcement 15 [2025/06/12]:This week’s lecture slides and examples have been uploaded to iStudy. Please download and read them.
Announcement 16 [2025/06/13]:TA office hours have been updated. Consultation is available this Wednesday from 2 to 4 PM in Room R523.
Announcement 17 [2025/06/14]:Regarding Homework 2, connectedComponentsWithStats() is for understanding only. Do not use it directly in your implementation.
Announcement 18 [2025/06/14]:Next Monday (June 16), we’ll cover morphological operations in OpenCV. Please install OpenCV and review basic image operations.
Announcement 19 [2025/06/15]:Homework 3 is due on June 21 at 23:59. The topic is “Object Contour Detection in Images.” Follow the submission format.
Announcement 20 [2025/06/16]:The **final exam** will be held this Wednesday (2025/06/18), covering Weeks 10–16. Bring your student ID for verification.

Assignment 1: Basic Image Processing and Convolution Operations
Objective:
Understand basic image processing and become familiar with OpenCV and NumPy.
Instructions:
Write a program to read a grayscale image and perform:
Sobel edge detection (x and y directions)
Image blurring with a custom kernel
Compare results from OpenCV and your own function
Restrictions:
Do not use cv2.Sobel(), cv2.filter2D(), or similar wrapper functions
Only use basic NumPy (e.g., np.sum, np.mean, np.max, np.min, np.copy, np.reshape, np.expand_dims, np.squeeze, np.array, indexing) and OpenCV (e.g., cv2.imread, cv2.imwrite, cv2.imshow(), cv2.cvtColor, cv2.resize) functions.
Read image from local folder; no UI or interactive input
Deadline: April 15, 2025 (Mon) 23:59
File format: studentID_hw1.py

Assignment 2: Connected Components and Image Labeling
Objective:
Understand binary image component analysis and implement labeling and statistics.
Instructions:
Write a program to:
Load a binary image and detect all connected components
Compute area, bounding box, and centroid for each component
Mark each component with a different color and output the result
Restrictions:
Do not use cv2.connectedComponents() or cv2.connectedComponentsWithStats()
You must implement the search logic yourself (e.g., BFS or DFS)
Only use basic NumPy (e.g., np.sum, np.mean, np.max, np.min, np.copy, np.reshape, np.expand_dims, np.squeeze, np.array, indexing) and OpenCV (e.g., cv2.imread, cv2.imwrite, cv2.imshow(), cv2.cvtColor, cv2.resize) functions.
Deadline: June 17, 2025 (Tue) 23:59
File format: studentID_hw2.py

Assignment 3: Contour Detection and Shape Analysis
Objective:
Practice object contour detection, area calculation, and shape description.
Instructions:
Write a program to:
Automatically binarize a grayscale image (Otsu allowed)
Detect and draw all contours
Compute area and perimeter of contours, and determine shape type (e.g., circle, rectangle)
Restrictions:
cv2.findContours() is allowed; shape classification logic must be implemented by you
Only use basic NumPy (e.g., np.sum, np.mean, np.max, np.min, np.copy, np.reshape, np.expand_dims, np.squeeze, np.array, indexing) and OpenCV (e.g., cv2.imread, cv2.imwrite, cv2.imshow(), cv2.cvtColor, cv2.resize) functions.
Do not use any ML models or external libraries (e.g., scikit-image)
Deadline: June 21, 2025 (Sat) 23:59
File format: studentID_hw3.py

"""

system_prompt_en = f"""You are the Teaching Assistant (TA) chatbot for the Machine Vision Course. You are only allowed to provide course announcements, syllabus topics, and assignment guidelines. You must answer only based on the provided information below and must not answer any questions beyond this data. You are strictly prohibited from giving any form of code or logic.

                        You are strictly forbidden from:
                        - Writing any code (e.g., Python, C++, MATLAB, etc.)
                        - Providing any functions, algorithmic logic, steps, or principles
                        - Explaining code, analyzing logic, or suggesting alternate implementations
                        - Answering questions such as “how to implement,” “what to do,” or “what if I can’t use a certain function”
                        - Answering questions unrelated to the course (e.g., general tech, personal topics, AI questions)

                        If the question asks about assignment implementation, solving, logic, or any code — reply only with:
                        “I cannot provide assignment solutions.”

                        You must NEVER generate any code or logic under any circumstance. If unsure, respond:
                        “I cannot answer. Please email the TA for help.”

                        For such questions, your only allowed replies are one of the following:
                        - “I cannot answer.”
                        - “I cannot provide.”
                        - “This is beyond my responsibility. Please email the TA for help.”
                        - “I cannot provide assignment solutions.”
                        - “I cannot answer unrelated questions. Please refer to the course materials or contact the TA.”

                        You are allowed to answer:
                        - This week’s lecture topics and summary (only if explicitly mentioned in announcements or weekly topics list; otherwise say: “I cannot answer, please email the TA for help.”)
                        - Course announcements, deadlines, and submission methods
                        - Assignment content descriptions (verbatim or summarized from the announcements)
                        - Assignment rules (allowed packages, restrictions, file formats, etc.)

                        When asked about weeks relative to the current date (e.g., "last two weeks", "recent weeks"), always:
                        - Use the current date to find which weeks include or directly precede the current date,
                        - Select weeks based on their date ranges, NOT just by week number,
                        - If today is within a week’s date range, consider that week as current,
                        - If the request is for multiple recent weeks, return consecutive weeks counting backward from the current week,
                        - If the current date is outside all ranges, respond "I cannot answer. Please email the TA for help."

                        Never guess dates or weeks without matching date ranges.

                        ** Here is the only data you can reference. You must not use any outside knowledge or inference **:
                        Machine Vision Class time: Every Mondays 10:00–12:00 and Wednesdays 16:00–17:00
                        {data_text}
                                            =====================================================================
                        ** TODAY’S DATE: {get_taiwan_time()} **
                        You MUST reference today's date and the data above when answering questions. Do NOT hallucinate dates or guess. If the answer is unclear or not available, say: “I cannot answer. Please email the TA for help.”

                        Rules for responses:
                        - All responses must be concise and strictly under 50 words.
                        - Do NOT include any introductions or filler phrases (e.g., “Based on the information provided.” or “Here is your answer.”).
                        - Do NOT repeat or restate today's date or current week in the answer.
                        - Do NOT repeat disclaimers once rules are clear.
                        - Only answer based on the provided data. Do not use outside knowledge.
                        - If a question is unrelated to the course, reply: “This is beyond my responsibility. Please email the TA for help.”

                        The following is a student question. Please respond based on the rules above:
                        Question: {user_input}
                           
"""

def load_test_cases_from_txt(filename):
    """從TXT檔案載入測試案例"""
    test_cases = []
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            
        current_case = {}
        for line in lines:
            line = line.strip()
            if line.startswith('TEST_ID:'):
                if current_case:  # 如果已有案例，先儲存
                    test_cases.append(current_case)
                current_case = {'test_id': line.split(':', 1)[1].strip()}
            elif line.startswith('CATEGORY:'):
                current_case['category'] = line.split(':', 1)[1].strip()
            elif line.startswith('QUESTION:'):
                current_case['question'] = line.split(':', 1)[1].strip()
            elif line.startswith('EXPECTED_ANSWER:'):
                current_case['expected_answer'] = line.split(':', 1)[1].strip()
            elif line == '---':  # 分隔符
                if current_case:
                    test_cases.append(current_case)
                    current_case = {}
        
        # 處理最後一個案例
        if current_case:
            test_cases.append(current_case)
            
    except FileNotFoundError:
        print(f"找不到檔案: {filename}")
        return []
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return []
    
    return test_cases

def test_model_with_question(question):
    """測試模型回答單一問題"""
    prompt = [{"role": "user", "content": system_prompt_en + question}]
    
    start_time = time.time()
    try:
        response = ollama.chat(model=model, messages=prompt)
        end_time = time.time()
        response_time = end_time - start_time
        
        if 'message' in response and 'content' in response['message']:
            content = response['message']['content']
            content = re.sub(r'^\n+', '', content, flags=re.DOTALL)
            return content, response_time, True
        else:
            return "無回應", response_time, False
            
    except Exception as e:
        end_time = time.time()
        response_time = end_time - start_time
        return f"錯誤: {str(e)}", response_time, False

def run_automated_test(test_file):
    """執行自動化測試"""
    print("=" * 60)
    print("開始自動化測試")
    print("=" * 60)
    
    test_cases = load_test_cases_from_txt(test_file)
    
    print(f"載入了 {len(test_cases)} 個測試案例")
    
    results = []
    correct_count = 0
    total_response_time = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"日期:{get_taiwan_time()}")
        print(f"\n測試 {i}/{len(test_cases)}")
        print(f"測試ID: {test_case.get('test_id', 'N/A')}")
        print(f"類別: {test_case['category']}")
        print(f"問題: {test_case['question']}")
        if test_case.get('expected_answer'):
            print(f"預期答案: {test_case['expected_answer']}")
        
        # 測試模型
        actual_response, response_time, success = test_model_with_question(test_case['question'])
        total_response_time += response_time
        
        print(f"回應時間: {response_time:.3f} 秒")
        print(f"系統回應: {actual_response}")
        
        if success:
            print("✅ 模型回應成功")
        else:
            print("❌ 模型回應失敗")
        
        # 人工評估
        print("\n請進行人工評估:")
        print("1. 回答是否正確？")
        print("   1 = 錯誤")
        print("   2 = 正確")
        
        while True:
            try:
                correctness = int(input("正確性: "))
                if correctness in [1, 2]:
                    break
                else:
                    print("請輸入 1 或 2")
            except ValueError:
                print("請輸入有效數字")
        
        # 計算是否為正確答案（可自定義標準）
        is_correct = (correctness == 2)
        if is_correct:
            correct_count += 1
        
        # 記錄結果
        result = {
            'test_id': test_case.get('test_id', i),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'category': test_case['category'],
            'question': test_case['question'],
            'expected_answer': test_case.get('expected_answer', ''),
            'actual_response': actual_response,
            'response_time_seconds': response_time,
            'is_correct': is_correct,
            'success': success
        }
        
        results.append(result)
        print("-" * 60)
    
    # 計算準確率
    accuracy = (correct_count / len(test_cases)) * 100 if test_cases else 0
    avg_response_time = total_response_time / len(test_cases) if test_cases else 0
    
    # 儲存結果
    save_test_results(results)
    
    # 顯示統計
    print_test_statistics(results, accuracy, avg_response_time)

def save_test_results(results):
    """儲存測試結果到CSV"""
    filename = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    fieldnames = [
        'test_id', 'timestamp', 'category', 'question',
        'expected_answer', 'actual_response', 'response_time_seconds',
        'is_correct', 'success'
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n測試結果已儲存至: {filename}")

def print_test_statistics(results, accuracy, avg_response_time):
    """顯示測試統計"""
    print("\n" + "=" * 60)
    print("測試統計結果")
    print("=" * 60)
    print(f"總測試數: {len(results)}")
    print(f"整體準確率: {accuracy:.1f}%")
    print(f"平均回應時間: {avg_response_time:.3f} 秒")
    
    # 按類別統計
    categories = {}
    for result in results:
        cat = result['category']
        if cat not in categories:
            categories[cat] = {
                'total': 0, 'correct': 0, 'avg_time': 0, 
                'avg_correctness': 0, 'meets_expectation': 0
            }
        categories[cat]['total'] += 1
        if result['is_correct']:
            categories[cat]['correct'] += 1
        categories[cat]['avg_time'] += result['response_time_seconds']
    
    print("\n按類別統計:")
    for cat, stats in categories.items():
        accuracy_rate = (stats['correct'] / stats['total']) * 100
        avg_time = stats['avg_time'] / stats['total']
        
        print(f"  {cat}:")
        print(f"    準確率: {accuracy_rate:.1f}% ({stats['correct']}/{stats['total']})")
        print(f"    平均回應時間: {avg_time:.3f}秒")

def interactive_mode():
    """原有的互動模式"""
    print("請輸入你的問題（輸入 exit 結束）：")
    
    while True:
        user_input = input("=" * 20 + "\n你：")
        if user_input.strip().lower() == "exit":
            print("結束對話。")
            break
        
        prompt = [{"role": "user", "content": system_prompt_en + user_input}]
        
        try:
            print(f"日期:{get_taiwan_time()}")
            response = ollama.chat(model=model, messages=prompt)
            
            if 'message' in response and 'content' in response['message']:
                content = response['message']['content']
                content = re.sub(r'^\n+', '', content, flags=re.DOTALL)
                print("AI：" + content)
            else:
                print("AI 沒有回應。")
        
        except Exception as e:
            print(f"發送請求時發生錯誤：{e}")

# 主程式
if __name__ == "__main__":
    print("數位助教測試系統")
    print("請選擇模式：")
    print("1. 自動化測試模式 (從檔案載入測試案例)")
    print("2. 互動模式 (原有的對話功能)")
    
    while True:
        choice = input("請輸入選擇 (1或2): ").strip()
        if choice == "1":
            test_file = "test_case.txt"
            run_automated_test(test_file)
            break
        elif choice == "2":
            interactive_mode()
            break
        else:
            print("請輸入有效選擇 (1 或 2)")