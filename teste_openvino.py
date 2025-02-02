from openvino import Core

ie = Core()
print("Dispositivos disponíveis:", ie.available_devices)  # Deve mostrar ['CPU', 'GPU']
    