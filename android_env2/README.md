## Installation


### Install Android Emulator
#### Windows
1. Install Java: Download and install Java from [here](https://www.oracle.com/java/technologies/downloads/). Make sure you set the JAVA_HOME environment variable. You can check if Java is installed correctly by running the command java --version in any command prompt window, which should display the installed Java version.
2. Install Android SDK Command line tools:
   - Download the [Command line tools](https://developer.android.com/studio) and extract them.
   - Move the extracted `cmdline-tools` directory to a new directory of your choice, for example, `android_sdk`. This new directory will be your Android SDK directory.
   - Inside the extracted `cmdline-tools directory`, create a new subdirectory named `latest`.
   - Move the contents of the original cmdline-tools directory (including the `lib` directory, `bin` directory, `NOTICE.txt` file, and `source.properties` file) to the newly created `latest` directory. Now, you can use the command line tools from this location. 
3. Install platform tools: Run the following command in the command prompt:
```
android_sdk\cmdline-tools\latest\bin\sdkmanager.bat "platform-tools" "platforms;android-33"
```
4. Download the Android image (API-level: 33):
```
android_sdk\cmdline-tools\latest\bin\sdkmanager.bat "system-images;android-33;google_apis_playstore;x86_64"
```
5. Create an Android Virtual Device (AVD):
```
android_sdk\cmdline-tools\latest\bin\avdmanager.bat create avd -n avd33 -k "system-images;android-33;google_apis_playstore;x86_64"
```
6. Launch the AVD:
   - For the Android GUI:
    ```
    android_sdk\emulator\emulator.exe -avd avd33 -memory 512 -partition-size 1024 -no-snapshot-load
    ```
   - For headless mode (no Android GUI):
    ```
    android_sdk\emulator\emulator.exe -avd avd33 -memory 512 -partition-size 1024 -no-snapshot-load
    ```
7. Test ADB connection:
```
android_sdk\platform-tools\adb.exe connect 127.0.0.1:5555
android_sdk\platform-tools\adb.exe devices
```
8. Run the following command to install the ATX application on the emulator:
```
python3 -m uiautomator2 init
```

#### Linux
The installation process for Linux is similar to Windows, with some additional steps:

1. Install Java and set the environment variables:
```
export JAVA_HOME=/home/user_name/java/jdk-xx.x.x.x  # Replace with your actual JDK installation directory
export JRE_HOME=${JAVA_HOME}/jre
export CLASSPATH=.:${JAVA_HOME}/lib:${JRE_HOME}/lib
export PATH=${JAVA_HOME}/bin:$PATH
```
2. Follow the same steps as Windows for installing Android SDK Command line tools, platform tools, and creating an AVD.
3. Launch the AVD:
   - For the Android GUI:
    ```
    android_sdk\emulator\emulator -avd avd33 -memory 512 -partition-size 1024 -no-snapshot-load
    ```
   - For headless mode (no Android GUI):
    ```
    android_sdk\emulator\emulator -avd avd33 -memory 512 -partition-size 1024 -no-snapshot-load
    ```
4. Test ADB connection:
```
android_sdk\platform-tools\adb connect 127.0.0.1:5555
android_sdk\platform-tools\adb devices
```
5. Run the following command to install the ATX application on the emulator:
```
python3 -m uiautomator2 init
```

#### Additional setup
1. Please mannuly setup your Google account
2. Turn off APP auto-upgrade in Google Play



### Troubleshoot
1. If you encounter the error "packaging.version.InvalidVersion: Invalid version: ''", you may need to enable uiautomator2 in the emulator:
   - On the emulator, open the ATX app
   - Click on "Start uiautomator"
2. Cannot `set_text` in TextView or EditView
Check `Settings` -> `System` -> `Language & Input` -> `Physical Keyboard` -> turn on `Use on-screen keyboard`
3. Black screen
https://www.cnblogs.com/yongdaimi/p/17464095.html
`android_sdk\emulator\emulator.exe -avd avd33 -memory 512 -partition-size 1024 -no-snapshot-load`