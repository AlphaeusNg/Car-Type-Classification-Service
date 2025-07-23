#!/usr/bin/env python3
"""
Car Type Classification Service - Run Script
Simplified setup and deployment script for Docker and uvicorn builds.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Configuration
PROJECT_NAME = "car-classification-service"
DOCKER_IMAGE = f"{PROJECT_NAME}:latest"
DEFAULT_PORT = 8000
API_MODULE = "api.main:app"

class Colors:
    """ANSI color codes for terminal output"""
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_banner():
    """Print application banner"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}🚗 Car Type Classification Service{Colors.END}")
    print(f"{Colors.BLUE}={'=' * 50}{Colors.END}\n")

def run_command(command, description, check=True):
    """Run a shell command with proper error handling"""
    print(f"{Colors.YELLOW}📋 {description}...{Colors.END}")
    print(f"{Colors.BLUE}   Command: {command}{Colors.END}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=True, 
            text=True
        )
        
        if result.stdout:
            print(f"{Colors.GREEN}   ✅ Success: {result.stdout.strip()}{Colors.END}")
        
        return result.returncode == 0
        
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}   ❌ Error: {e.stderr.strip()}{Colors.END}")
        if check:
            sys.exit(1)
        return False

def check_requirements(skip_docker=False):
    """Check if required tools are installed"""
    print(f"{Colors.BOLD}🔍 Checking requirements...{Colors.END}")
    
    requirements = {
        'python3': 'python3 --version',
        'pip': 'pip --version'
    }
    
    if not skip_docker:
        requirements['docker'] = 'docker --version'
    
    missing = []
    for tool, command in requirements.items():
        if not run_command(command, f"Checking {tool}", check=False):
            missing.append(tool)
    
    if missing:
        if 'docker' in missing and skip_docker:
            print(f"{Colors.YELLOW}⚠️  Docker not available, skipping Docker functionality{Colors.END}")
        else:
            print(f"{Colors.RED}❌ Missing requirements: {', '.join(missing)}{Colors.END}")
            if 'docker' in missing:
                print(f"{Colors.YELLOW}💡 Install Docker with: sudo apt update && sudo apt install docker.io{Colors.END}")
                print(f"{Colors.YELLOW}💡 Or run with --skip-docker to use local mode only{Colors.END}")
            print(f"{Colors.YELLOW}💡 Please install missing tools and try again{Colors.END}")
            sys.exit(1)
    
    print(f"{Colors.GREEN}✅ All requirements satisfied{Colors.END}\n")
    return True

def setup_environment():
    """Set up Python virtual environment"""
    print(f"{Colors.BOLD}🐍 Setting up Python environment...{Colors.END}")
    
    venv_path = Path('.venv')
    
    if not venv_path.exists():
        run_command('python3 -m venv .venv', 'Creating virtual environment')
    else:
        print(f"{Colors.GREEN}   ✅ Virtual environment already exists{Colors.END}")
    
    # Activate environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = '.venv\\Scripts\\activate'
        pip_cmd = '.venv\\Scripts\\pip'
    else:  # Linux/macOS
        activate_cmd = 'source .venv/bin/activate'
        pip_cmd = '.venv/bin/pip'
    
    run_command(f'{pip_cmd} install --upgrade pip', 'Upgrading pip')
    run_command(f'{pip_cmd} install -r requirements.txt', 'Installing dependencies')
    
    print(f"{Colors.GREEN}✅ Python environment ready{Colors.END}\n")

def build_docker():
    """Build Docker image"""
    print(f"{Colors.BOLD}🐳 Building Docker image...{Colors.END}")
    
    # Check if model files exist
    model_files = ['car_classification_model.h5', 'class_mapping.json']
    missing_files = [f for f in model_files if not Path(f).exists()]
    
    if missing_files:
        print(f"{Colors.YELLOW}⚠️  Warning: Missing model files: {', '.join(missing_files)}{Colors.END}")
        print(f"{Colors.YELLOW}   Please run the training notebook first to generate model files{Colors.END}")
        
        response = input(f"{Colors.YELLOW}   Continue building Docker image? (y/N): {Colors.END}")
        if response.lower() != 'y':
            print(f"{Colors.YELLOW}   Skipping Docker build{Colors.END}\n")
            return False
    
    # Build Docker image
    build_cmd = f'sudo docker build -t {DOCKER_IMAGE} .'
    success = run_command(build_cmd, 'Building Docker image')
    
    if success:
        print(f"{Colors.GREEN}✅ Docker image built successfully: {DOCKER_IMAGE}{Colors.END}\n")
    
    return success

def run_local_api(port=DEFAULT_PORT, reload=True):
    """Run API locally with uvicorn"""
    print(f"{Colors.BOLD}🚀 Starting local API server...{Colors.END}")
    
    # Check if model files exist
    model_files = ['car_classification_model.h5', 'class_mapping.json']
    missing_files = [f for f in model_files if not Path(f).exists()]
    
    if missing_files:
        print(f"{Colors.RED}❌ Missing required model files: {', '.join(missing_files)}{Colors.END}")
        print(f"{Colors.YELLOW}💡 Please run the training notebook first to generate model files{Colors.END}")
        return False
    
    # Activate virtual environment and run uvicorn
    if os.name == 'nt':  # Windows
        python_cmd = '.venv\\Scripts\\python'
    else:  # Linux/macOS
        python_cmd = '.venv/bin/python'
    
    reload_flag = '--reload' if reload else ''
    uvicorn_cmd = f'{python_cmd} -m uvicorn {API_MODULE} --host 0.0.0.0 --port {port} {reload_flag}'
    
    print(f"{Colors.GREEN}🌐 API will be available at: http://localhost:{port}{Colors.END}")
    print(f"{Colors.GREEN}📚 API docs available at: http://localhost:{port}/docs{Colors.END}")
    print(f"{Colors.YELLOW}Press Ctrl+C to stop the server{Colors.END}\n")
    
    try:
        subprocess.run(uvicorn_cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Server stopped by user{Colors.END}")
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}❌ Failed to start server: {e}{Colors.END}")
        return False
    
    return True

def run_docker_api(port=DEFAULT_PORT):
    """Run API in Docker container"""
    print(f"{Colors.BOLD}🐳 Starting Docker container...{Colors.END}")
    
    # Check if Docker image exists
    check_cmd = f'sudo docker images -q {DOCKER_IMAGE}'
    result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
    
    if not result.stdout.strip():
        print(f"{Colors.YELLOW}⚠️  Docker image not found. Building first...{Colors.END}")
        if not build_docker():
            return False
    
    # Stop any existing container
    stop_cmd = f'sudo docker stop {PROJECT_NAME} 2>/dev/null || true'
    remove_cmd = f'sudo docker rm {PROJECT_NAME} 2>/dev/null || true'
    subprocess.run(stop_cmd, shell=True, capture_output=True)
    subprocess.run(remove_cmd, shell=True, capture_output=True)
    
    # Run new container
    docker_cmd = f'sudo docker run --name {PROJECT_NAME} -p {port}:{DEFAULT_PORT} {DOCKER_IMAGE}'
    
    print(f"{Colors.GREEN}🌐 API will be available at: http://localhost:{port}{Colors.END}")
    print(f"{Colors.GREEN}📚 API docs available at: http://localhost:{port}/docs{Colors.END}")
    print(f"{Colors.YELLOW}Press Ctrl+C to stop the container{Colors.END}\n")
    
    try:
        subprocess.run(docker_cmd, shell=True, check=True)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Stopping container...{Colors.END}")
        subprocess.run(f'sudo docker stop {PROJECT_NAME}', shell=True, capture_output=True)
        subprocess.run(f'sudo docker rm {PROJECT_NAME}', shell=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"{Colors.RED}❌ Failed to run Docker container: {e}{Colors.END}")
        return False
    
    return True

def setup_project(skip_docker=False):
    """Complete project setup"""
    print(f"{Colors.BOLD}🔧 Setting up Car Type Classification Service...{Colors.END}")
    
    check_requirements(skip_docker=skip_docker)
    setup_environment()
    
    # Try to build Docker image if Docker is available
    if not skip_docker:
        try:
            print(f"{Colors.BOLD}🐳 Preparing Docker environment...{Colors.END}")
            build_docker()
        except Exception as e:
            print(f"{Colors.YELLOW}⚠️  Docker setup failed: {e}{Colors.END}")
            print(f"{Colors.YELLOW}   You can still use local mode{Colors.END}")
    
    print(f"{Colors.GREEN}✅ Project setup complete!{Colors.END}")
    print(f"{Colors.YELLOW}💡 Next steps:{Colors.END}")
    print(f"   1. Run the training notebook: jupyter notebook model_training.ipynb")
    print(f"   2. Start the API: python run.py --mode local")
    print(f"   3. Test the API: curl http://localhost:8000/health\n")

def auto_setup_and_run():
    """Automatically set up the project and run the API"""
    print(f"{Colors.BOLD}🚀 Auto Setup & Run Mode{Colors.END}")
    print(f"{Colors.BLUE}This will automatically:{Colors.END}")
    print(f"  1. Check requirements")
    print(f"  2. Set up Python environment") 
    print(f"  3. Install dependencies")
    print(f"  4. Check for trained models")
    print(f"  5. Start the API server\n")
    
    # Check if Docker is available
    docker_available = subprocess.run('docker --version', shell=True, capture_output=True).returncode == 0
    
    # Setup environment
    check_requirements(skip_docker=not docker_available)
    setup_environment()
    
    # Check for model files
    model_files = ['car_classification_model.h5', 'class_mapping.json']
    missing_files = [f for f in model_files if not Path(f).exists()]
    
    if missing_files:
        print(f"{Colors.YELLOW}⚠️  Missing model files: {', '.join(missing_files)}{Colors.END}")
        print(f"{Colors.YELLOW}   Please run the training notebook first:{Colors.END}")
        print(f"{Colors.BLUE}   jupyter notebook model_training.ipynb{Colors.END}")
        
        response = input(f"{Colors.YELLOW}   Continue without models (API will fail)? (y/N): {Colors.END}")
        if response.lower() != 'y':
            print(f"{Colors.YELLOW}   Please train the model first, then run again{Colors.END}")
            return False
    
    # Choose deployment method
    if docker_available and Path('car_classification_model.h5').exists():
        print(f"{Colors.BOLD}🎯 Choosing deployment method...{Colors.END}")
        response = input(f"{Colors.YELLOW}Run in Docker (d) or Local (l)? [l]: {Colors.END}")
        
        if response.lower() == 'd':
            build_docker()
            run_docker_api()
        else:
            run_local_api()
    else:
        if not docker_available:
            print(f"{Colors.YELLOW}💡 Docker not available, running locally{Colors.END}")
        run_local_api()
    
    return True

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Car Type Classification Service - Run Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                            # Auto setup and run (recommended)
  python run.py --setup                    # Setup project environment only
  python run.py --mode local               # Run API locally with uvicorn
  python run.py --mode docker              # Run API in Docker container
  python run.py --build                    # Build Docker image only
  python run.py --mode local --port 8080   # Run on custom port
        """
    )
    
    parser.add_argument('--mode', choices=['local', 'docker'], 
                       help='Deployment mode: local (uvicorn) or docker')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                       help=f'Port to run the API (default: {DEFAULT_PORT})')
    parser.add_argument('--setup', action='store_true',
                       help='Setup project environment only')
    parser.add_argument('--build', action='store_true',
                       help='Build Docker image only')
    parser.add_argument('--no-reload', action='store_true',
                       help='Disable auto-reload for local mode')
    parser.add_argument('--skip-docker', action='store_true',
                       help='Skip Docker checks and functionality')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Handle different modes
    if args.setup:
        setup_project(skip_docker=args.skip_docker)
    elif args.build:
        check_requirements(skip_docker=False)
        build_docker()
    elif args.mode == 'local':
        check_requirements(skip_docker=True)
        setup_environment()
        run_local_api(port=args.port, reload=not args.no_reload)
    elif args.mode == 'docker':
        check_requirements(skip_docker=False)
        run_docker_api(port=args.port)
    else:
        # Auto setup and run mode (default)
        auto_setup_and_run()

if __name__ == '__main__':
    main()
