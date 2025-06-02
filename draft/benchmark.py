import requests
import time
from datetime import datetime
import statistics

# Test data
test_data = {
    "markdown": "# Meeting Notes\n\n**Date:** [Insert Date]  \n**Time:** [Insert Time]  \n**Location:** [Insert Location]  \n**Attendees:** [List of Attendees]\n\n## Agenda\n\n1. [Agenda Item 1]\n2. [Agenda Item 2]\n3. [Agenda Item 3]\n\n## Discussion Points\n\n- **Topic 1:**  \n  - Key Point 1  \n  - Key Point 2\n\n- **Topic 2:**  \n  - Key Point 1  \n  - Key Point 2\n\n- **Topic 3:**  \n  - Key Point 1  \n  - Key Point 2\n\n## Action Items\n\n1. **Action Item 1:** [Description]  \n   - Responsible: [Name]  \n   - Deadline: [Date]\n\n2. **Action Item 2:** [Description]  \n   - Responsible: [Name]  \n   - Deadline: [Date]\n\n## Next Meeting\n\n- **Date:** [Insert Date]  \n- **Time:** [Insert Time]  \n- **Location:** [Insert Location]\n\n## Additional Notes\n\n- [Any additional notes or comments]",
    "style_id": 8
}

# Endpoints to test
endpoints = {
    "VM Endpoint": "https://vm.synthesis.tricklau.xyz/convert",
    "Standard Endpoint": "https://synthesis.tricklau.xyz/convert"
}

def run_benchmark(num_requests=5):
    results = {}
    
    for endpoint_name, url in endpoints.items():
        print(f"\nTesting {endpoint_name}...")
        response_times = []
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                response = requests.post(url, json=test_data)
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                response_times.append(response_time)
                print(f"Request {i+1}: {response_time:.2f}ms")
                
                if response.status_code != 200:
                    print(f"Error: Status code {response.status_code}")
                    print(response.text)
            except Exception as e:
                print(f"Error during request {i+1}: {str(e)}")
                continue
        
        if response_times:
            results[endpoint_name] = {
                "min": min(response_times),
                "max": max(response_times),
                "avg": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
            }
    
    return results

def generate_markdown_report(results):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    markdown = f"""# Handwriting Synthesis API Benchmark Report
Generated on: {timestamp}

## Test Configuration
- Number of requests per endpoint: 10
- Test data: Meeting notes template
- Style ID: 8

## Results

| Endpoint | Min (ms) | Max (ms) | Average (ms) | Median (ms) | Std Dev (ms) |
|----------|----------|----------|--------------|-------------|--------------|
"""
    
    for endpoint_name, metrics in results.items():
        markdown += f"| {endpoint_name} | {metrics['min']:.2f} | {metrics['max']:.2f} | {metrics['avg']:.2f} | {metrics['median']:.2f} | {metrics['std_dev']:.2f} |\n"
    
    # Add performance comparison
    if len(results) == 2:
        endpoints = list(results.keys())
        avg_diff = results[endpoints[1]]['avg'] - results[endpoints[0]]['avg']
        markdown += f"\n## Performance Comparison\n"
        markdown += f"- {endpoints[1]} is {abs(avg_diff):.2f}ms {'slower' if avg_diff > 0 else 'faster'} than {endpoints[0]}\n"
        markdown += f"- Performance difference: {abs(avg_diff/results[endpoints[0]]['avg']*100):.2f}%\n"
    
    return markdown

def main():
    print("Starting benchmark...")
    results = run_benchmark()
    
    # Generate and save report
    report = generate_markdown_report(results)
    with open("benchmark_report.md", "w") as f:
        f.write(report)
    
    print("\nBenchmark completed! Report saved to benchmark_report.md")

if __name__ == "__main__":
    main() 