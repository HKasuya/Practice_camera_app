import cv2
import sys

# 顔検出の準備（検出）ローカルcv2フォルダ、dataの中にある
face_cascade = cv2.CascadeClassifier(
    "CV2をインストールすると次のパスが作られるのでこの絶対パスを指定。/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_frontalface_alt.xml"
)
eye_cascade = cv2.CascadeClassifier(
    "CV2をインストールすると次のパスが作られるのでこの絶対パスを指定。/Local/Programs/Python/Python313/Lib/site-packages/cv2/data/haarcascade_eye_tree_eyeglasses.xml"
)

# Pcの備え付けカメラを使用（０は備え付け、他にカメラがある場合は１や２も使う）
cap = cv2.VideoCapture(0)
# 止め方のコメント表示
print("escを押したら止まります。")

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

    # 検出速度向上のためグレースケール変換
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔検出開始
    faces = face_cascade.detectMultiScale(gray_image)

    # 検出された顔にバウンディングボックスを表示
    for x, y, w, h in faces:
        # 顔の位置に四角を描画
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        """
        x:検出された顔の左上隅のx座標です。画像の左端からの水平方向の距離を示します。
        y: 検出された顔の左上隅のy座標です。画像の上端からの垂直方向の距離を示します。
        w: 検出された顔の幅（width）です。顔の水平方向のサイズを示します。
        h: 検出された顔の高さ（height）です。顔の垂直方向のサイズを示します。
        (0,255,0)はRBGの色
        最後の１は線の太さ　デフォルトは１
        """

        # 目の検出開始（グレースケール画像 gray_image の中で、以前に検出された顔が検出された領域だけを切り出している）
        eyes = eye_cascade.detectMultiScale(gray_image[y : y + h, x : x + w])
        # 目の位置に四角を描画
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(
                frame[y : y + h, x : x + w],
                (ex, ey),
                (ex + ew, ey + eh),
                (255, 255, 0),
                1,
            )

    # 読み込んだフレームを画面に表示
    cv2.imshow("カメラアプリ(バウンディングボックス）)", frame)

    # escキーを押したら止まる
    key = cv2.waitKey(1)
    if key == 27:  # 27はasciiではESCキーをさす
        break

# キャプチャを開放してウィンドウを閉じる
cap.release()  # cap.release()を呼び出さないと、ビデオソースが解放されないため、プログラム終了後にカメラが使用できなくなる可能性がある。
cv2.destroyAllWindows
