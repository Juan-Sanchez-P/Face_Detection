import eel

eel.init('Gui')

@eel.expose
def App():
    print('Application Runnung')

App()

eel.start('index.html', size=(500, 600))