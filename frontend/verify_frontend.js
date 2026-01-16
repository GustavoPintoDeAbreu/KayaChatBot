/**
 * Simple syntax check for React components
 * Run with: node verify_frontend.js
 */

const fs = require('fs');
const path = require('path');

console.log('=' .repeat(60));
console.log('Frontend Verification');
console.log('='.repeat(60));

const checks = {
    passed: 0,
    failed: 0,
    warnings: 0
};

// Check if files exist
const requiredFiles = [
    'package.json',
    'vite.config.js',
    'index.html',
    'src/main.jsx',
    'src/App.jsx',
    'src/App.css',
    'src/index.css',
    'Dockerfile',
    'nginx.conf'
];

console.log('\nChecking required files...');
requiredFiles.forEach(file => {
    const filePath = path.join(__dirname, file);
    if (fs.existsSync(filePath)) {
        console.log(`✓ ${file}`);
        checks.passed++;
    } else {
        console.log(`✗ ${file} - MISSING`);
        checks.failed++;
    }
});

// Check package.json dependencies
console.log('\nChecking package.json dependencies...');
try {
    const packageJson = JSON.parse(fs.readFileSync('package.json', 'utf-8'));
    
    const requiredDeps = ['react', 'react-dom', 'axios', 'lucide-react'];
    const requiredDevDeps = ['vite', '@vitejs/plugin-react'];
    
    requiredDeps.forEach(dep => {
        if (packageJson.dependencies && packageJson.dependencies[dep]) {
            console.log(`✓ ${dep}`);
            checks.passed++;
        } else {
            console.log(`✗ ${dep} - MISSING`);
            checks.failed++;
        }
    });
    
    requiredDevDeps.forEach(dep => {
        if (packageJson.devDependencies && packageJson.devDependencies[dep]) {
            console.log(`✓ ${dep} (dev)`);
            checks.passed++;
        } else {
            console.log(`✗ ${dep} (dev) - MISSING`);
            checks.failed++;
        }
    });
    
    // Check scripts
    console.log('\nChecking npm scripts...');
    const requiredScripts = ['dev', 'build', 'preview'];
    requiredScripts.forEach(script => {
        if (packageJson.scripts && packageJson.scripts[script]) {
            console.log(`✓ ${script} script`);
            checks.passed++;
        } else {
            console.log(`✗ ${script} script - MISSING`);
            checks.failed++;
        }
    });
    
} catch (e) {
    console.log(`✗ Error reading package.json: ${e.message}`);
    checks.failed++;
}

// Check React component structure
console.log('\nChecking React component structure...');
try {
    const appContent = fs.readFileSync('src/App.jsx', 'utf-8');
    
    // Check for essential React patterns
    const patterns = [
        { name: 'React import', pattern: /import\s+React/ },
        { name: 'useState hook', pattern: /useState/ },
        { name: 'useEffect hook', pattern: /useEffect/ },
        { name: 'axios import', pattern: /import\s+axios/ },
        { name: 'Login component', pattern: /LoginRequest|handleLogin/ },
        { name: 'Chat component', pattern: /handleSendMessage/ },
        { name: 'Export default', pattern: /export\s+default/ }
    ];
    
    patterns.forEach(({ name, pattern }) => {
        if (pattern.test(appContent)) {
            console.log(`✓ ${name}`);
            checks.passed++;
        } else {
            console.log(`⚠ ${name} - Not found (might be okay)`);
            checks.warnings++;
        }
    });
    
} catch (e) {
    console.log(`✗ Error checking App.jsx: ${e.message}`);
    checks.failed++;
}

// Check Vite config
console.log('\nChecking Vite configuration...');
try {
    const viteConfig = fs.readFileSync('vite.config.js', 'utf-8');
    
    const vitePatterns = [
        { name: 'React plugin', pattern: /@vitejs\/plugin-react/ },
        { name: 'Server config', pattern: /server:/ },
        { name: 'Proxy config', pattern: /proxy:/ },
        { name: 'Preview config', pattern: /preview:/ }
    ];
    
    vitePatterns.forEach(({ name, pattern }) => {
        if (pattern.test(viteConfig)) {
            console.log(`✓ ${name}`);
            checks.passed++;
        } else {
            console.log(`⚠ ${name} - Not found`);
            checks.warnings++;
        }
    });
    
} catch (e) {
    console.log(`✗ Error checking vite.config.js: ${e.message}`);
    checks.failed++;
}

// Summary
console.log('\n' + '='.repeat(60));
console.log('Verification Summary');
console.log('='.repeat(60));
console.log(`✓ Passed: ${checks.passed}`);
console.log(`✗ Failed: ${checks.failed}`);
console.log(`⚠ Warnings: ${checks.warnings}`);

if (checks.failed === 0) {
    console.log('\n🎉 Frontend structure looks good!');
    console.log('Next steps:');
    console.log('  1. Run: npm install');
    console.log('  2. Run: npm run dev (for development)');
    console.log('  3. Run: npm run build (for production)');
} else {
    console.log('\n⚠ Some issues found. Please review the output above.');
}

process.exit(checks.failed === 0 ? 0 : 1);
