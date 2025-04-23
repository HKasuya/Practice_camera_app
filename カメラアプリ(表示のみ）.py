import cv2 as cv
import sys

cap = cv.VideoCapture(0)

print("escキーで終了します。")

# もしカメラが開かなかたら
if not cap.isOpened():
    print("カメラを開けません")
    # プログラムを終了
    sys.exit()

# カメラを継続的に表示させる
while True:
    # カメラから画像を読み込む
    ret, frame = cap.read()
    # 画像が正しく読み込めたか確認
    if not ret:
        print("フレームの読み込みに失敗しました。")
        # while文なのでループを抜けるため（プログラムは止まらない）
        break

    # 読み込んだフレームを画面に表示
    cv.imshow("カメラアプリ", frame)
    # escキーが押されたらループ終了
    key = cv.waitKey(1)
    if key == 27:  # 27はasciiではESCキーをさす
        break

# キャプチャを開放してウィンドウを閉じる
cap.release()
cv.destroyAllWindows
