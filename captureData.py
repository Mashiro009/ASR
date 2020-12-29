import pyaudio
import wave
from tqdm import tqdm
import time

def record_audio(wave_out_path,record_second):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)


    wf = wave.open(wave_out_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)


    print("* recording")

    for i in tqdm(range(0, int(RATE / CHUNK * record_second))):
        data = stream.read(CHUNK)
        wf.writeframes(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()

    p.terminate()

    wf.close()



def play():
    # 用文本文件记录wave模块解码每一帧所产生的内容。注意这里不是保存为二进制文件
    dump_buff_file = open(r"Ring01.dup", 'w')

    chunk = 1  # 指定WAV文件的大小
    wf = wave.open(r"output.wav", 'rb')  # 打开WAV文件
    p = pyaudio.PyAudio()  # 初始化PyAudio模块

    # 打开一个数据流对象，解码而成的帧将直接通过它播放出来，我们就能听到声音啦
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(),
                    rate=wf.getframerate(), output=True)

    data = wf.readframes(chunk)  # 读取第一帧数据
    print(data)  # 以文本形式打印出第一帧数据，实际上是转义之后的十六进制字符串

    # 播放音频，并使用while循环继续读取并播放后面的帧数
    # 结束的标志为wave模块读到了空的帧
    while data != b'':
        stream.write(data)  # 将帧写入数据流对象中，以此播放之
        data = wf.readframes(chunk)  # 继续读取后面的帧
        dump_buff_file.write(str(data) + "\n---------------------------------------\n")  # 将读出的帧写入文件中，每一个帧用分割线隔开以便阅读

    stream.stop_stream()  # 停止数据流
    stream.close()  # 关闭数据流
    p.terminate()  # 关闭 PyAudio
    print('play函数结束！')

def play_audio(wave_path):

    CHUNK = 1024

    wf = wave.open(wave_path, 'rb')

    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()

    # open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data
    data = wf.readframes(CHUNK)

    # play stream (3)
    datas = []
    while len(data) > 0:
        data = wf.readframes(CHUNK)
        datas.append(data)

    for d in tqdm(datas):
        stream.write(d)

    # stop stream (4)
    stream.stop_stream()
    stream.close()

    # close PyAudio (5)
    p.terminate()


def play_audio_callback(wave_path):
  CHUNK = 1024
  wf = wave.open(wave_path, 'rb')
  # instantiate PyAudio (1)
  p = pyaudio.PyAudio()
  def callback(in_data, frame_count, time_info, status):
    data = wf.readframes(frame_count)
    return (data, pyaudio.paContinue)
  # open stream (2)
  stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
          channels=wf.getnchannels(),
          rate=wf.getframerate(),
          output=True,
          stream_callback=callback)
  # read data
  stream.start_stream()
  while stream.is_active():
    time.sleep(0.1)
  # stop stream (4)
  stream.stop_stream()
  stream.close()
  # close PyAudio (5)
  p.terminate()

# play()
record_audio("output.wav",record_second=10)
play_audio("output.wav")
# play_audio_callback("output.wav")