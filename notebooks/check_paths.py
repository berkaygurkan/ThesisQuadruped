import os

# Aramaya başlayacağımız ana dizin (Notebooks'un bir üstü)
SEARCH_ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))

print(f"--- Arama Başlıyor: {SEARCH_ROOT} ---\n")

found_logs = []

# Tüm alt klasörleri gez
for root, dirs, files in os.walk(SEARCH_ROOT):
    for file in files:
        if "events.out.tfevents" in file:
            full_path = root
            # notebooks klasörüne göre relative path hesapla (Script orada çalışacağı için)
            # Ama garanti olsun diye absolute path (tam yol) kullanacağız.
            print(f"[BULUNDU] {full_path}")
            found_logs.append(full_path)

print("\n" + "="*50)
print("Aşağıdaki LOG_DIRS bloğunu kopyalayıp plot koduna yapıştırın:")
print("="*50)

print("LOG_DIRS = {")
for i, path in enumerate(found_logs):
    # Klasör ismini etiket olarak kullanalım (Daha sonra değiştirebilirsin)
    folder_name = os.path.basename(path)
    # Eğer PPO_1 ise bir üst klasör adını alalım ki anlamlı olsun
    if folder_name == "PPO_1":
        folder_name = os.path.basename(os.path.dirname(path))
        
    print(f'    "Model_{i}_{folder_name}": r"{path}",')
print("}")
print("="*50)