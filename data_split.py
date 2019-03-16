# coding=utf-8
import os
import io
import random
import json

path = '/data/CWS/Action_Recognition_using_Visual_Attention/data/UCF-101/'
labels = {'JugglingBalls': 0, 'RopeClimbing': 1, 'WritingOnBoard': 2, 'PoleVault': 3, 'Punch': 4, 'WallPushups': 5, 'FrisbeeCatch': 6, 'PullUps': 7, 'BlowDryHair': 8, 'TennisSwing': 9, 'Hammering': 10, 'SkateBoarding': 11, 'Nunchucks': 12, 'VolleyballSpiking': 13, 'Rafting': 14, 'PushUps': 15, 'SkyDiving': 16, 'FieldHockeyPenalty': 17, 'BrushingTeeth': 18, 'MilitaryParade': 19, 'Fencing': 20, 'Typing': 21, 'Skijet': 22, 'BoxingSpeedBag': 23, 'ShavingBeard': 24, 'CricketShot': 25, 'FrontCrawl': 26, 'ThrowDiscus': 27, 'Shotput': 28, 'Mixing': 29, 'HandstandPushups': 30, 'BlowingCandles': 31, 'BreastStroke': 32, 'BaseballPitch': 33, 'SalsaSpin': 34, 'BandMarching': 35, 'Haircut': 36, 'BodyWeightSquats': 37, 'HorseRace': 38, 'PommelHorse': 39, 'JavelinThrow': 40, 'UnevenBars': 41, 'MoppingFloor': 42, 'FloorGymnastics': 43, 'GolfSwing': 44, 'RockClimbingIndoor': 45, 'Basketball': 46, 'PlayingFlute': 47, 'HighJump': 48, 'BasketballDunk': 49, 'TableTennisShot': 50, 'Knitting': 51, 'PlayingSitar': 52, 'ApplyLipstick': 53, 'Biking': 54, 'HandstandWalking': 55, 'CuttingInKitchen': 56, 'HulaHoop': 57, 'YoYo': 58, 'PlayingGuitar': 59, 'Surfing': 60, 'Swing': 61, 'BabyCrawling': 62, 'PlayingTabla': 63, 'TrampolineJumping': 64, 'SoccerJuggling': 65, 'JumpingJack': 66, 'PlayingCello': 67, 'CleanAndJerk': 68, 'PlayingDhol': 69, 'ApplyEyeMakeup': 70, 'Archery': 71, 'PlayingPiano': 72, 'StillRings': 73, 'Drumming': 74, 'PlayingViolin': 75, 'TaiChi': 76, 'BoxingPunchingBag': 77, 'IceDancing': 78, 'Lunges': 79, 'PizzaTossing': 80, 'Billiards': 81, 'Kayaking': 82, 'Skiing': 83, 'HammerThrow': 84, 'BenchPress': 85, 'SumoWrestling': 86, 'CliffDiving': 87, 'LongJump': 88, 'Diving': 89, 'CricketBowling': 90, 'Rowing': 91, 'HorseRiding': 92, 'ParallelBars': 93, 'PlayingDaf': 94, 'BalanceBeam': 95, 'HeadMassage': 96, 'JumpRope': 97, 'SoccerPenalty': 98, 'Bowling': 99, 'WalkingWithDog': 100}

if not os.path.exists('data/train.json'):
    fobj = open('data/train.json', 'w')
    fobj.close()

if not os.path.exists('data/test.json'):
    fobj = open('data/test.json', 'w')
    fobj.close()

trainSet = []
testSet = []
for ele in labels:
    label = labels[ele]
    videoList = os.listdir(path+ele)
    videoListLen = len(videoList)
    testLen = int(videoListLen*0.3)
    trainLen = videoListLen - testLen
    testIdx = [random.randint(0,videoListLen-1) for _ in range(testLen)]
    trainIdx = []
    for i in range(videoListLen):
        flag = True
        for j in testIdx:
            if i == j:
                flag = False
        if flag == True:
            trainIdx.append(i)
    for idx in trainIdx:
        trainData = {}
        trainData['video_name'] = videoList[idx]
        trainData['label'] = label
        trainSet.append(trainData)

    for idx in testIdx:
        testData = {}
        testData['video_name'] = videoList[idx]
        testData['label'] = label
        testSet.append(testData)
random.shuffle(trainSet)
random.shuffle(testSet)
with open('data/train.json', 'w') as f:
    json.dump(trainSet, f)
with open('data/val.json', 'w') as f:
    json.dump(testSet, f)
print('Done!')
