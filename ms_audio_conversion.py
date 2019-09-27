from pydub import AudioSegment
sound = AudioSegment.from_mp3("/home/muthumurugan/Desktop/GoldenAudio/muthu2/attachments/muthu_51sec_mobile_gionee_earphone.mp3")
sound.export("/home/muthumurugan/Desktop/GoldenAudio/muthu_converted/muthu_51sec_mobile_gionee_earphone.wav", format="wav")
