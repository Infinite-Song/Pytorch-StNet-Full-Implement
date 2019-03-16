# coding = utf-8
import os
import  subprocess

labels = {'JugglingBalls': 0, 'RopeClimbing': 1, 'WritingOnBoard': 2, 'PoleVault': 3, 'Punch': 4, 'WallPushups': 5, 'FrisbeeCatch': 6, 'PullUps': 7, 'BlowDryHair': 8, 'TennisSwing': 9, 'Hammering': 10, 'SkateBoarding': 11, 'Nunchucks': 12, 'VolleyballSpiking': 13, 'Rafting': 14, 'PushUps': 15, 'SkyDiving': 16, 'FieldHockeyPenalty': 17, 'BrushingTeeth': 18, 'MilitaryParade': 19, 'Fencing': 20, 'Typing': 21, 'Skijet': 22, 'BoxingSpeedBag': 23, 'ShavingBeard': 24, 'CricketShot': 25, 'FrontCrawl': 26, 'ThrowDiscus': 27, 'Shotput': 28, 'Mixing': 29, 'HandstandPushups': 30, 'BlowingCandles': 31, 'BreastStroke': 32, 'BaseballPitch': 33, 'SalsaSpin': 34, 'BandMarching': 35, 'Haircut': 36, 'BodyWeightSquats': 37, 'HorseRace': 38, 'PommelHorse': 39, 'JavelinThrow': 40, 'UnevenBars': 41, 'MoppingFloor': 42, 'FloorGymnastics': 43, 'GolfSwing': 44, 'RockClimbingIndoor': 45, 'Basketball': 46, 'PlayingFlute': 47, 'HighJump': 48, 'BasketballDunk': 49, 'TableTennisShot': 50, 'Knitting': 51, 'PlayingSitar': 52, 'ApplyLipstick': 53, 'Biking': 54, 'HandstandWalking': 55, 'CuttingInKitchen': 56, 'HulaHoop': 57, 'YoYo': 58, 'PlayingGuitar': 59, 'Surfing': 60, 'Swing': 61, 'BabyCrawling': 62, 'PlayingTabla': 63, 'TrampolineJumping': 64, 'SoccerJuggling': 65, 'JumpingJack': 66, 'PlayingCello': 67, 'CleanAndJerk': 68, 'PlayingDhol': 69, 'ApplyEyeMakeup': 70, 'Archery': 71, 'PlayingPiano': 72, 'StillRings': 73, 'Drumming': 74, 'PlayingViolin': 75, 'TaiChi': 76, 'BoxingPunchingBag': 77, 'IceDancing': 78, 'Lunges': 79, 'PizzaTossing': 80, 'Billiards': 81, 'Kayaking': 82, 'Skiing': 83, 'HammerThrow': 84, 'BenchPress': 85, 'SumoWrestling': 86, 'CliffDiving': 87, 'LongJump': 88, 'Diving': 89, 'CricketBowling': 90, 'Rowing': 91, 'HorseRiding': 92, 'ParallelBars': 93, 'PlayingDaf': 94, 'BalanceBeam': 95, 'HeadMassage': 96, 'JumpRope': 97, 'SoccerPenalty': 98, 'Bowling': 99, 'WalkingWithDog': 100}

def getLength(filename):    #get the duration of the video
    result = subprocess.Popen(['ffprobe', filename],
      stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    string=[x for x in result.stdout.readlines() if 'Duration' in x]
    time=int(string[0][18:20])
    return time

def video2image(rpath, spath):
    # '-r ' + str(n)
    time = getLength(rpath)+1
    n = int(35/time)+1
    strcmd = 'ffmpeg -i ' + '"' + rpath + '"' + ' -r ' + str(n) + ' -s 224*224 -f image2 ' + '"' + spath + '%d.jpg"'
    subprocess.call(strcmd, shell=True)

for ele in labels:
    path = '/data/CWS/Action_Recognition_using_Visual_Attention/data/UCF-101/' + ele
    videoList = os.listdir(path)
    for video in videoList:
        rpath = path + '/' + video
        spath = 'data/ucf/' + video[0:-4] + '/'
        if not os.path.exists(spath):
            os.makedirs(spath)
        video2image(rpath, spath)
        img_list = os.listdir(spath)
        img_num = len(img_list)
        if img_num < 35:
            for num in range(img_num+1, 36):
                cpcmd = 'cp ' + spath + str(img_num) + '.jpg ' + spath + str(num) + '.jpg'
                subprocess.call(cpcmd, shell=True)
                num += 1
        if img_num > 35:
            for num in range(36, img_num+1):
                decmd = 'rm ' + spath + str(num) + '.jpg'
                subprocess.call(decmd, shell=True)
                num += 1