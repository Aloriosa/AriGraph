import os
import json

def summarize_results():
    """Summarize results from both tasks"""
    results = {}
    
    # Inpainting results
    inpainting_dir = 'results/inpainting'
    if os.path.exists(os.path.join(inpainting_dir, 'metrics.txt')):
        with open(os.path.join(inpainting_dir, 'metrics.txt'), 'r') as f:
            lines = f.readlines()
            psnr = float(lines[0].split(':')[1].strip())
            fid = float(lines[1].split(':')[1].strip())
            results['inpainting'] = {'PSNR': psnr, 'FID': fid}
    
    # Super-resolution results
    super_res_dir = 'results/super_resolution'
    if os.path.exists(os.path.join(super_res_dir, 'metrics.txt')):
        with open(os.path.join(super_res_dir, 'metrics.txt'), 'r') as f:
            lines = f.readlines()
            psnr = float(lines[0].split(':')[1].strip())
            fid = float(lines[1].split(':')[1].strip())
            results['super_resolution'] = {'PSNR': psnr, 'FID': fid}
    
    # Print summary
    print("\n" + "="*50)
    print("REPRODUCTION SUMMARY")
    print("="*50)
    
    if 'inpainting' in results:
        print(f"Inpainting - PSNR: {results['inpainting']['PSNR']:.2f}, FID: {results['inpainting']['FID']:.2f}")
    else:
        print("Inpainting - Results not available")
    
    if 'super_resolution' in results:
        print(f"Super-Resolution - PSNR: {results['super_resolution']['PSNR']:.2f}, FID: {results['super_resolution']['FID']:.2f}")
    else:
        print("Super-Resolution - Results not available")
    
    # Save summary
    with open('results/summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSummary saved to results/summary.json")

if __name__ == '__main__':
    summarize_results()