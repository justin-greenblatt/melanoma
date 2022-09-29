from subprocess import Popen, PIPE
import os
from settings import config as homelessConfig

#configure home Directorie
homelessConfig["paths"]["home"] = os.environ.get("HOME")
configOut = open('/' + str(os.path.join(*os.path.realpath(__file__).split('/')[:-1], "settings", "config.ini")), 'w')
homelessConfig.write(configOut)
configOut.close()

from settings import config, sConfig

def runCommand(command):
    p = Popen(command, stdout = PIPE, stdin = PIPE)
    print(" ".join(command))
    p.wait()
    print("finished")
    return p.communicate()

for p in sConfig["apt"].values():
    runCommand(["sudo", "apt", "install", p, "-y"])
for q in sConfig["pip"].values():
    runCommand(["sudo", "pip3", "install", q, "--no-input"])

#formatDisk
formatDiskCommand = ["sudo", "mkfs.ext4", "-m", "0", "-E", "lazy_itable_init=0,lazy_journal_init=0,discard", "/dev/sdb"]
runCommand(formatDiskCommand)

print("-----formatedDisk-----")

def createDir(dirName):
    if not os.path.isdir(dirName):
        runCommand(["sudo","mkdir", dirName])

createDir(config["paths"]["data"])
createDir(config["paths"]["kaggle"])

mountDiskCommand = ["sudo", "mount", "-o", "discard,defaults", "/dev/sdb", config["paths"]["data"]]
runCommand(mountDiskCommand)

print("-----mounted disk-----")

runCommand(["sudo","cp",os.path.join(config["paths"]["home"],"melanoma","kaggle.json"),
          os.path.join(config["paths"]["kaggle"],"kaggle.json")])
runCommand(["sudo","chmod", "777", "-R", config["paths"]["data"]])
os.chdir(config["paths"]["data"])

downloadCommand = ["kaggle", "competitions", "download", "-c", sConfig["dataset"]["kaggle_competition"]]

runCommand(downloadCommand)
runCommand(["unzip", sConfig["dataset"]["kaggle_competition"] + ".zip"])


for d in config["paths"].values():
    createDir(d)
    print("---created directorie : "+d)

