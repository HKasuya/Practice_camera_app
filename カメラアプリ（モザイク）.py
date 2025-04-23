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

    # モザイク
    def mosaic(frame, rect, size):
        x, y, w, h = rect
        # res = cv.resize(img,(width, height), interpolation = cv.INTER_CUBIC)を使用。INTER_CUBICはリサイズの補完方法の１つ
        small = cv2.resize(
            frame[y : y + h, x : x + w], (size, size), interpolation=cv2.INTER_LINEAR
        )  # cv2.INTER_LINEARも補完方法の１つで、よく使われるもの

        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    """
    w,hは元の縦横（サイズ）
    cv2.INTER_NEAREST は最近傍補間と呼ばれる方法で、
    拡大時に最も近いピクセルの値をそのままコピーして新しいピクセルを生成。
    この補間方法を使うと、画像が滑らかにならず、元の小さなピクセルのブロックがそのまま拡大されるため、
    粗いモザイクのような効果が得られる。
    """

    mosaic_rate = 10  # モザイクの荒さ調整のパラメータ　変数名は基本このまま使用される
    # 顔の位置を検出して座標へ
    for x, y, w, h in faces:
        mosaic_region = mosaic(frame, (x, y, w, h), mosaic_rate)
        frame[y : y + h, x : x + w] = (
            mosaic_region  # モザイクを元の顔の領域に入れてあげる
        )

    # 読み込んだフレームを画面に表示
    cv2.imshow("カメラアプリ(モザイク）)", frame)

    # escキーを押したら止まる
    key = cv2.waitKey(1)
    if key == 27:  # 27はasciiではESCキーをさす
        break

# キャプチャを開放してウィンドウを閉じる
cap.release()
cv2.destroyAllWindows
