#!/usr/bin/env python3
"""
Codebase Analysis for Render Deployment
Analyzes the 555-serverlite system codebase metrics
"""

import os
import glob

def analyze_codebase():
    py_files = glob.glob('*.py')
    total_lines = 0
    file_details = []

    print('📊 CODEBASE ANALYSIS for Render Deployment')
    print('=' * 50)

    for file_path in py_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
            file_size = os.path.getsize(file_path) / 1024
            filename = os.path.basename(file_path)
            file_details.append({'file': filename, 'lines': lines, 'size': file_size})
            total_lines += lines
            print(f'{filename:<30} {lines:>8} lines  {file_size:>6.1f} KB')
        except Exception as e:
            print(f'Error reading {file_path}: {e}')

    file_details.sort(key=lambda x: x['lines'], reverse=True)
    main_file = file_details[0] if file_details else {'file': 'N/A', 'lines': 0}
    total_size = sum(f['size'] for f in file_details)

    print()
    print('🎯 DEPLOYMENT SUMMARY:')
    print(f'📁 Total Python files: {len(py_files)}')
    print(f'📝 Total lines of code: {total_lines:,}')
    print(f'💾 Total codebase size: {total_size:.1f} KB')
    print()
    print('🚀 RENDER DEPLOYMENT CONSIDERATIONS:')
    print(f'• Main server: {main_file["file"]} ({main_file["lines"]:,} lines)')
    
    if total_lines > 10000:
        complexity = 'High'
        recommendations = ['Consider code splitting', 'Monitor memory usage', 'Use efficient imports']
    elif total_lines > 5000:
        complexity = 'Medium'  
        recommendations = ['Good for deployment', 'Monitor performance']
    else:
        complexity = 'Low'
        recommendations = ['Optimized for deployment']

    print(f'• Complexity level: {complexity}')
    print(f'• Memory footprint: {"Optimized" if total_size < 500 else "Standard"}')
    
    print()
    print('💡 RECOMMENDATIONS:')
    for rec in recommendations:
        print(f'  - {rec}')
    
    # Check for deployment files
    deployment_files = ['requirements.txt', 'Procfile', 'render.yaml', '.env']
    print()
    print('🔧 DEPLOYMENT FILES CHECK:')
    for dep_file in deployment_files:
        status = "✅ Found" if os.path.exists(dep_file) else "❌ Missing"
        print(f'  {dep_file:<20} {status}')

if __name__ == "__main__":
    analyze_codebase()