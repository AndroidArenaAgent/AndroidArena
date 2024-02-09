import time

import uiautomator2 as u2

def setup_emulator(d):
    # set home page for firefox
    # TODO setup for firefox
    d.app_start('org.mozilla.firefox', use_monkey=True)
    d.xpath('//*[@resource-id="org.mozilla.firefox:id/menuButton"]').click()
    d.xpath(
        '//*[@resource-id="org.mozilla.firefox:id/mozac_browser_menu_recyclerView"]/android.widget.LinearLayout[8]').click()
    d.xpath(
        '//*[@resource-id="org.mozilla.firefox:id/recycler_view"]/android.widget.LinearLayout[3]/android.widget.RelativeLayout[1]').click()
    d.swipe_ext("up")
    d.xpath('//*[@resource-id="org.mozilla.firefox:id/recycler_view"]/android.view.ViewGroup[3]').click()

    # set auto-sync for Photo
    d.app_start('com.google.android.apps.photos', use_monkey=True)
    d.xpath('//*[@resource-id="com.google.android.apps.photos:id/og_apd_internal_image_view"]').click()
    d.xpath('//*[@resource-id="com.google.android.apps.photos:id/photos_autobackup_particle_generic_button"]').click()
    d.xpath('//*[@resource-id="com.google.android.apps.photos:id/done_button"]').click()
    
def push_files(d):
    d.push("scripts/prepare_files/sample.pdf", "/sdcard/Download/", show_progress=True)
    d.push("scripts/prepare_files/sample1.pdf", "/sdcard/Download/", show_progress=True)
    d.push("scripts/prepare_files/image.jpg", "/sdcard/Download/", show_progress=True)
    d.push("scripts/prepare_files/image1.jpeg", "/sdcard/Download/", show_progress=True)


def install_apps(d):
    # install IBM Weather, Slack and Firefox
    d.open_url("https://play.google.com/store/apps/details?id=com.weather.Weather")
    d.xpath('//*[@content-desc="Install"]').click()
    time.sleep(60)
    d.open_url("https://play.google.com/store/apps/details?id=org.mozilla.firefox")
    d.xpath('//*[@content-desc="Install"]').click()
    time.sleep(60)
    d.open_url("https://play.google.com/store/apps/details?id=com.Slack")
    d.xpath('//*[@content-desc="Install"]').click()
    time.sleep(60)


def create_contacts(d):
    d.app_start('com.google.android.contacts', use_monkey=True)
    d.xpath('//*[@resource-id="com.google.android.contacts:id/floating_action_button"]').click()
    d.xpath(
        '//*[@resource-id="com.google.android.contacts:id/kind_section_views"]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.FrameLayout[1]').set_text(
        'John')
    d.xpath(
        '//*[@resource-id="com.google.android.contacts:id/kind_section_views"]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[2]/android.widget.FrameLayout[1]').set_text(
        'Smith')
    d.swipe_ext('up')
    d.xpath(
        '//*[@resource-id="com.google.android.contacts:id/kind_section_views"]/android.widget.LinearLayout[3]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.FrameLayout[1]').set_text(
        '010-123456')
    d.swipe_ext('up')
    d.xpath('//*[@resource-id="com.google.android.contacts:id/more_fields"]').click()
    d.swipe_ext('up')
    d.swipe_ext('up')
    d.xpath(
        '//*[@resource-id="com.google.android.contacts:id/kind_section_views"]/android.widget.LinearLayout[3]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.FrameLayout[1]').set_text(
        'Mountain View, CA 94045')
    d.app_stop('com.google.android.contacts')

    d.app_start('com.google.android.contacts', use_monkey=True)
    d.xpath('//*[@resource-id="com.google.android.contacts:id/floating_action_button"]').click()
    d.xpath(
        '//*[@resource-id="com.google.android.contacts:id/kind_section_views"]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.FrameLayout[1]').set_text(
        'Bob')
    d.xpath(
        '//*[@resource-id="com.google.android.contacts:id/kind_section_views"]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[2]/android.widget.FrameLayout[1]').set_text(
        'Steve')
    d.swipe_ext('up')
    d.xpath(
        '//*[@resource-id="com.google.android.contacts:id/kind_section_views"]/android.widget.LinearLayout[3]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.LinearLayout[1]/android.widget.FrameLayout[1]').set_text(
        '010-321456')
    d.xpath('//*[@resource-id="com.google.android.contacts:id/menu_save"]').click()


if __name__ == "__main__":
    device = u2.connect("emulator-5554")
    install_apps(device)
    push_files(device)
    create_contacts(device)
