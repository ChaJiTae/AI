import speech_recognition as sr

# 동작 리스트
# menu_list = ["초기 화면", "직원 호출", "결제", "주문","취소","아메리카노"]

# 메뉴 리스트
menu_list = ["카푸치노","아메리카노","에소프레소","주문 취소"]

def recognize_speech_from_mic():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("주문을 말씀해주세요!")
        audio = recognizer.listen(source)
    try:
        # 구글 음성 인식의 언어 : 한국어
        response = recognizer.recognize_google(audio, language="ko-KR")
        print(f"인식된 음성: {response}")
        return response.lower()
    except sr.RequestError:
        print("API를 사용할 수 없거나 응답이 없습니다.")
        return None
    except sr.UnknownValueError:
        print(f"음성을 인식할 수 없습니다."  )
        return None

# 동작 일치 여부
def match_menu_order(recognized_text):
    for menu_item in menu_list:
        if menu_item in recognized_text:
            return menu_item
    return "일치하는 메뉴 항목이 없습니다."

# 음성 주문 인식 및 매칭
if __name__ == "__main__":
    recognized_text = recognize_speech_from_mic()
    if recognized_text:
        selected_menu = match_menu_order(recognized_text)
        print(f"선택된 메뉴 항목: {selected_menu}")
    else:
        print("음성을 인식하지 못했거나 음성 입력이 없습니다.")
