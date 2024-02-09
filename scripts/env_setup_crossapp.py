import time

import uiautomator2 as u2


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
    d.xpath('//*[@resource-id="com.google.android.contacts:id/menu_save"]').click()


def setup_emulator_crossapp(d):
    # gmail draft
    d.app_start('com.google.android.gm', use_monkey=True)
    d.xpath('//*[@resource-id="com.google.android.gm:id/compose_button"]').click()
    d.xpath('//*[@resource-id="com.google.android.gm:id/subject"]').set_text('OpenAI website')
    d.xpath('//*[@text="Compose email"]').set_text('https://openai.com/')
    d.press('enter')
    d.xpath('//*[@content-desc="Navigate up"]').click()
    time.sleep(3)

    d.xpath('//*[@resource-id="com.google.android.gm:id/compose_button"]').click()
    d.xpath('//*[@resource-id="com.google.android.gm:id/subject"]').set_text('meeting details')
    d.xpath('//*[@text="Compose email"]').set_text('meeting at 13:00')
    d.press('enter')
    d.xpath('//*[@content-desc="Navigate up"]').click()
    time.sleep(3)
    # d.xpath('//*[@resource-id="com.google.android.gm:id/conversation_container"]/android.webkit.WebView[1]/android.webkit.WebView[1]').set_text('weekly meeting on 13 next month, Microsoft SVC Building')

    d.xpath('//*[@resource-id="com.google.android.gm:id/compose_button"]').click()
    d.xpath('//*[@resource-id="com.google.android.gm:id/subject"]').set_text('restaurant reservation')
    d.xpath('//*[@text="Compose email"]').set_text('3 kingdoms hotpot')
    d.press('enter')
    d.xpath('//*[@content-desc="Navigate up"]').click()
    time.sleep(3)

    d.xpath('//*[@resource-id="com.google.android.gm:id/compose_button"]').click()
    d.xpath('//*[@resource-id="com.google.android.gm:id/subject"]').set_text('flight confirmation')
    d.xpath('//*[@text="Compose email"]').set_text('Columbia Metropolitan Airport')
    d.xpath('//*[@content-desc="Navigate up"]').click()
    time.sleep(3)

    d.xpath('//*[@resource-id="com.google.android.gm:id/compose_button"]').click()
    d.xpath('//*[@resource-id="com.google.android.gm:id/subject"]').set_text('YouTube video recommendation')
    d.xpath('//*[@text="Compose email"]').set_text('ChatGPT Explained Completely')
    d.xpath('//*[@content-desc="Navigate up"]').click()
    time.sleep(3)

    d.xpath('//*[@resource-id="com.google.android.gm:id/compose_button"]').click()
    d.xpath('//*[@resource-id="com.google.android.gm:id/subject"]').set_text('YouTube channel subscription')
    d.xpath('//*[@text="Compose email"]').set_text('trailer of movie The Godfather')
    d.xpath('//*[@content-desc="Navigate up"]').click()
    time.sleep(3)

    d.xpath('//*[@resource-id="com.google.android.gm:id/compose_button"]').click()
    d.xpath('//*[@resource-id="com.google.android.gm:id/subject"]').set_text('YouTube channel subscription')
    d.xpath('//*[@text="Compose email"]').set_text('trailer of movie The Godfather')
    d.xpath('//*[@content-desc="Navigate up"]').click()
    time.sleep(3)

    # Message
    d.app_start('com.google.android.apps.messaging', use_monkey=True)
    d.xpath('//*[@resource-id="com.google.android.apps.messaging:id/start_chat_fab"]').click()
    d.xpath('//android.widget.ScrollView').set_text('123')
    d.press('enter')
    d.xpath('//*[@resource-id="com.google.android.apps.messaging:id/compose_message_text"]').set_text(
        'weekly meeting on 13 Oct, Room 101')
    d.xpath('//*[@resource-id="com.google.android.apps.messaging:id/send_message_button_container"]').click()
    d.xpath('//*[@content-desc="Navigate up"]')


if __name__ == "__main__":
    device = u2.connect("emulator-5554")
    create_contacts(device)
    setup_emulator_crossapp(device)
